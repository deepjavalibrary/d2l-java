/*
 * Copyright 2021 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.modality.nlp.DefaultVocabulary;
import ai.djl.modality.nlp.Vocabulary;
import ai.djl.modality.nlp.embedding.TrainableWordEmbedding;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.core.Linear;
import ai.djl.nn.recurrent.GRU;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.dataset.Batch;
import ai.djl.training.initializer.XavierInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.PairList;
import java.io.IOException;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

public class JupyterSeq2SeqLearning {

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();
        ParameterStore ps = new ParameterStore(manager, false);

        Seq2SeqEncoder encoder = new Seq2SeqEncoder(10, 8, 16, 2, 0);
        NDArray X = manager.zeros(new Shape(4, 7));
        encoder.initialize(manager, DataType.FLOAT32, X.getShape());
        NDList outputState = encoder.forward(ps, new NDList(X), false);

        NDArray output = outputState.head();
        System.out.println(output.getShape());

        NDList state = outputState.subNDList(1);
        System.out.println(state.size());
        System.out.println(state.head().getShape());

        Seq2SeqDecoder decoder = new Seq2SeqDecoder(10, 8, 16, 2, 0);
        state = decoder.initState(outputState);
        NDList input = new NDList(X).addAll(state);
        decoder.initialize(manager, DataType.FLOAT32, input.getShapes());
        outputState = decoder.forward(ps, input, false);

        output = outputState.head();
        System.out.println(output.getShape());

        state = outputState.subNDList(1);
        System.out.println(state.size());
        System.out.println(state.head().getShape());

        X = manager.create(new int[][] {{1, 2, 3}, {4, 5, 6}});
        System.out.println(X.sequenceMask(manager.create(new int[] {1, 2})));

        X = manager.ones(new Shape(2, 3, 4));
        System.out.println(X.sequenceMask(manager.create(new int[] {1, 2}), -1));

        Loss loss = new MaskedSoftmaxCELoss();
        NDList labels = new NDList(manager.ones(new Shape(3, 4)));
        labels.add(manager.create(new int[] {4, 2, 0}));
        NDList predictions = new NDList(manager.ones(new Shape(3, 4, 10)));
        System.out.println(loss.evaluate(labels, predictions));
    }

    public static void trainSeq2Seq(
            EncoderDecoder net,
            ArrayDataset dataset,
            float lr,
            int numEpochs,
            Vocab tgtVocab,
            Device device)
            throws IOException, TranslateException {
        Loss loss = new MaskedSoftmaxCELoss();
        Tracker lrt = Tracker.fixed(lr);
        Optimizer adam = Optimizer.adam().optLearningRateTracker(lrt).build();

        DefaultTrainingConfig config =
                new DefaultTrainingConfig(loss)
                        .optOptimizer(adam) // Optimizer (loss function)
                        .optInitializer(new XavierInitializer(), "");

        Model model = Model.newInstance("seq2seq");
        model.setBlock(net);
        Trainer trainer = model.newTrainer(config);

        Animator animator = new Animator();
        StopWatch watch;
        Accumulator metric;
        double lossValue = 0, speed = 0;
        for (int epoch = 1; epoch <= numEpochs; epoch++) {
            watch = new StopWatch();
            metric = new Accumulator(2); // Sum of training loss, no. of tokens
            try (NDManager childManager = model.getNDManager().newSubManager(device)) {
                // Iterate over dataset
                ParameterStore ps = new ParameterStore(childManager, false);
                for (Batch batch : dataset.getData(childManager)) {
                    NDArray X = batch.getData().get(0);
                    NDArray lenX = batch.getData().get(1);
                    NDArray Y = batch.getLabels().get(0);
                    NDArray lenY = batch.getLabels().get(1);

                    NDArray bos =
                            childManager
                                    .full(new Shape(Y.getShape().get(0)), tgtVocab.getIdx("<bos>"))
                                    .reshape(-1, 1);
                    NDArray decInput =
                            NDArrays.concat(
                                    new NDList(bos, Y.get(new NDIndex(":, :-1"))),
                                    1); // Teacher forcing
                    try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                        NDArray yHat = net.forward(ps, new NDList(X, decInput, lenX), true).get(0);
                        NDArray l = loss.evaluate(new NDList(Y, lenY), new NDList(yHat));
                        gc.backward(l);
                        metric.add(new float[] {l.sum().getFloat(), lenY.sum().getLong()});
                    }
                    TrainingChapter9.gradClipping(net, 1, childManager);
                    // Update parameters
                    trainer.step();
                }
            }
            lossValue = metric.get(0) / metric.get(1);
            speed = metric.get(1) / watch.stop();
            if ((epoch + 1) % 10 == 0) {
                animator.add(epoch + 1, (float) lossValue, "loss");
                animator.show();
            }
        }
        System.out.format(
                "loss: %.3f, %.1f tokens/sec on %s%n", lossValue, speed, device.toString());
    }

    public static class Seq2SeqEncoder extends Encoder {

        private TrainableWordEmbedding embedding;
        private GRU rnn;

        /* The RNN encoder for sequence to sequence learning. */
        public Seq2SeqEncoder(
                int vocabSize, int embedSize, int numHiddens, int numLayers, float dropout) {
            List<String> list =
                    IntStream.range(0, vocabSize)
                            .mapToObj(String::valueOf)
                            .collect(Collectors.toList());
            Vocabulary vocab = new DefaultVocabulary(list);
            // Embedding layer
            embedding =
                    TrainableWordEmbedding.builder()
                            .optNumEmbeddings(vocabSize)
                            .setEmbeddingSize(embedSize)
                            .setVocabulary(vocab)
                            .build();
            addChildBlock("embedding", embedding);
            rnn =
                    GRU.builder()
                            .setNumLayers(numLayers)
                            .setStateSize(numHiddens)
                            .optReturnState(true)
                            .optBatchFirst(false)
                            .optDropRate(dropout)
                            .build();
            addChildBlock("rnn", rnn);
        }

        /** {@inheritDoc} */
        @Override
        public void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {
            embedding.initialize(manager, dataType, inputShapes[0]);
            Shape[] shapes = embedding.getOutputShapes(new Shape[] {inputShapes[0]});
            try (NDManager sub = manager.newSubManager()) {
                NDArray nd = sub.zeros(shapes[0], dataType);
                nd = nd.swapAxes(0, 1);
                rnn.initialize(manager, dataType, nd.getShape());
            }
        }

        @Override
        protected NDList forwardInternal(
                ParameterStore ps,
                NDList inputs,
                boolean training,
                PairList<String, Object> params) {
            NDArray X = inputs.head();
            // The output `X` shape: (`batchSize`, `numSteps`, `embedSize`)
            X = embedding.forward(ps, new NDList(X), training, params).head();
            // In RNN models, the first axis corresponds to time steps
            X = X.swapAxes(0, 1);

            return rnn.forward(ps, new NDList(X), training);
        }
    }

    public static class Seq2SeqDecoder extends Decoder {

        private TrainableWordEmbedding embedding;
        private GRU rnn;
        private Linear dense;

        /* The RNN decoder for sequence to sequence learning. */
        public Seq2SeqDecoder(
                int vocabSize, int embedSize, int numHiddens, int numLayers, float dropout) {
            List<String> list =
                    IntStream.range(0, vocabSize)
                            .mapToObj(String::valueOf)
                            .collect(Collectors.toList());
            Vocabulary vocab = new DefaultVocabulary(list);
            embedding =
                    TrainableWordEmbedding.builder()
                            .optNumEmbeddings(vocabSize)
                            .setEmbeddingSize(embedSize)
                            .setVocabulary(vocab)
                            .build();
            addChildBlock("embedding", embedding);
            rnn =
                    GRU.builder()
                            .setNumLayers(numLayers)
                            .setStateSize(numHiddens)
                            .optReturnState(true)
                            .optBatchFirst(false)
                            .optDropRate(dropout)
                            .build();
            addChildBlock("rnn", rnn);
            dense = Linear.builder().setUnits(vocabSize).build();
            addChildBlock("dense", dense);
        }

        /** {@inheritDoc} */
        @Override
        public void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {
            embedding.initialize(manager, dataType, inputShapes[0]);
            try (NDManager sub = manager.newSubManager()) {
                Shape shape = embedding.getOutputShapes(new Shape[] {inputShapes[0]})[0];
                NDArray nd = sub.zeros(shape, dataType).swapAxes(0, 1);
                NDArray state = sub.zeros(inputShapes[1], dataType);
                NDArray context = state.get(new NDIndex(-1));
                context =
                        context.broadcast(
                                new Shape(
                                        nd.getShape().head(),
                                        context.getShape().head(),
                                        context.getShape().get(1)));
                // Broadcast `context` so it has the same `numSteps` as `X`
                NDArray xAndContext = NDArrays.concat(new NDList(nd, context), 2);
                rnn.initialize(manager, dataType, xAndContext.getShape());
                shape = rnn.getOutputShapes(new Shape[] {xAndContext.getShape()})[0];
                dense.initialize(manager, dataType, shape);
            }
        }

        public NDList initState(NDList encOutputs) {
            return new NDList(encOutputs.get(1));
        }

        @Override
        protected NDList forwardInternal(
                ParameterStore parameterStore,
                NDList inputs,
                boolean training,
                PairList<String, Object> params) {
            NDArray X = inputs.head();
            NDArray state = inputs.get(1);
            X =
                    embedding
                            .forward(parameterStore, new NDList(X), training, params)
                            .head()
                            .swapAxes(0, 1);
            NDArray context = state.get(new NDIndex(-1));
            // Broadcast `context` so it has the same `numSteps` as `X`
            context =
                    context.broadcast(
                            new Shape(
                                    X.getShape().head(),
                                    context.getShape().head(),
                                    context.getShape().get(1)));
            NDArray xAndContext = NDArrays.concat(new NDList(X, context), 2);
            NDList rnnOutput =
                    rnn.forward(parameterStore, new NDList(xAndContext, state), training);
            NDArray output = rnnOutput.head();
            state = rnnOutput.get(1);
            output =
                    dense.forward(parameterStore, new NDList(output), training)
                            .head()
                            .swapAxes(0, 1);
            return new NDList(output, state);
        }
    }

    public static class MaskedSoftmaxCELoss extends SoftmaxCrossEntropyLoss {
        /* The softmax cross-entropy loss with masks. */

        @Override
        public NDArray evaluate(NDList labels, NDList predictions) {
            NDArray weights = labels.head().onesLike().expandDims(-1).sequenceMask(labels.get(1));
            // Remove the states from the labels NDList because otherwise, it will throw an error as
            // SoftmaxCrossEntropyLoss
            // expects only one NDArray for label and one NDArray for prediction
            labels.remove(1);
            return super.evaluate(labels, predictions).mul(weights).mean(new int[] {1});
        }
    }
}
