import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.dataset.Batch;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.loss.SoftmaxCrossEntropyLoss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Function;

public class TimeMachine {

    /** Split text lines into word or character tokens. */
    public static String[][] tokenize(String[] lines, String token) {
        String[][] output = new String[lines.length][];
        if ("word".equals(token)) {
            for (int i = 0; i < output.length; i++) {
                output[i] = lines[i].split(" ");
            }
        } else if ("char".equals(token)) {
            for (int i = 0; i < output.length; i++) {
                output[i] = lines[i].split("");
            }
        } else {
            throw new IllegalArgumentException("ERROR: unknown token type: " + token);
        }
        return output;
    }

    /** Read `The Time Machine` dataset and return an array of the lines */
    public static String[] readTimeMachine() throws IOException {
        URL url = new URL("http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt");
        String[] lines;
        try (BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()))) {
            lines = in.lines().toArray(String[]::new);
        }

        for (int i = 0; i < lines.length; i++) {
            lines[i] = lines[i].replaceAll("[^A-Za-z]+", " ").trim().toLowerCase();
        }
        return lines;
    }

    /** Return token indices and the vocabulary of the time machine dataset. */
    public static Pair<List<Integer>, Vocab> loadCorpusTimeMachine(int maxTokens)
            throws IOException {
        String[] lines = readTimeMachine();
        String[][] tokens = tokenize(lines, "char");
        Vocab vocab = new Vocab(tokens, 0, new String[0]);
        // Since each text line in the time machine dataset is not necessarily a
        // sentence or a paragraph, flatten all the text lines into a single list
        List<Integer> corpus = new ArrayList<>();
        for (String[] token : tokens) {
            for (String s : token) {
                if (!s.isEmpty()) {
                    corpus.add(vocab.getIdx(s));
                }
            }
        }
        if (maxTokens > 0) {
            corpus = corpus.subList(0, maxTokens);
        }
        return new Pair<>(corpus, vocab);
    }

    /** Generate new characters following the `prefix`. */
    public static String predictCh8(
            String prefix,
            int numPreds,
            Object net,
            Vocab vocab,
            Device device,
            NDManager manager) {

        List<Integer> outputs = new ArrayList<>();
        outputs.add(vocab.getIdx("" + prefix.charAt(0)));
        Functions.SimpleFunction<NDArray> getInput =
                () ->
                        manager.create(outputs.get(outputs.size() - 1))
                                .toDevice(device, false)
                                .reshape(new Shape(1, 1));

        if (net instanceof RNNModelScratch) {
            RNNModelScratch castedNet = (RNNModelScratch) net;
            NDList state = castedNet.beginState(1, device);

            for (char c : prefix.substring(1).toCharArray()) { // Warm-up period
                state = castedNet.forward(getInput.apply(), state).getValue();
                outputs.add(vocab.getIdx("" + c));
            }

            NDArray y;
            for (int i = 0; i < numPreds; i++) {
                Pair<NDArray, NDList> pair = castedNet.forward(getInput.apply(), state);
                y = pair.getKey();
                state = pair.getValue();

                outputs.add((int) y.argMax(1).reshape(new Shape(1)).getLong(0L));
            }
        } else {
            AbstractBlock castedNet = (AbstractBlock) net;
            NDList state = null;
            for (char c : prefix.substring(1).toCharArray()) { // Warm-up period
                ParameterStore ps = new ParameterStore(manager, false);
                NDList input = new NDList(getInput.apply());
                if (state == null) {
                    // Begin state
                    state = castedNet.forward(ps, input, false).subNDList(1);
                } else {
                    state = castedNet.forward(ps, input.addAll(state), false).subNDList(1);
                }
                outputs.add(vocab.getIdx("" + c));
            }

            NDArray y;
            for (int i = 0; i < numPreds; i++) {
                ParameterStore ps = new ParameterStore(manager, false);
                NDList input = new NDList(getInput.apply());
                NDList pair = castedNet.forward(ps, input.addAll(state), false);
                y = pair.get(0);
                state = pair.subNDList(1);

                outputs.add((int) y.argMax(1).reshape(new Shape(1)).getLong(0L));
            }
        }

        StringBuilder output = new StringBuilder();
        for (int i : outputs) {
            output.append(vocab.idxToToken.get(i));
        }
        return output.toString();
    }

    /** Train a model. */
    public static void trainCh8(
            Object net,
            RandomAccessDataset dataset,
            Vocab vocab,
            int lr,
            int numEpochs,
            Device device,
            boolean useRandomIter,
            NDManager manager)
            throws IOException, TranslateException {
        SoftmaxCrossEntropyLoss loss = new SoftmaxCrossEntropyLoss();
        Animator animator = new Animator();

        Functions.voidTwoFunction<Integer, NDManager> updater;
        if (net instanceof RNNModelScratch) {
            RNNModelScratch castedNet = (RNNModelScratch) net;
            updater =
                    (batchSize, subManager) ->
                            Training.sgd(castedNet.params, lr, batchSize, subManager);
        } else {
            // Already initialized net
            AbstractBlock castedNet = (AbstractBlock) net;
            Model model = Model.newInstance("model");
            model.setBlock(castedNet);

            Tracker lrt = Tracker.fixed(lr);
            Optimizer sgd = Optimizer.sgd().setLearningRateTracker(lrt).build();

            DefaultTrainingConfig config =
                    new DefaultTrainingConfig(loss)
                            .optOptimizer(sgd) // Optimizer (loss function)
                            .optInitializer(
                                    new NormalInitializer(0.01f),
                                    Parameter.Type.WEIGHT) // setting the initializer
                            .optDevices(
                                    manager.getEngine()
                                            .getDevices(1)) // setting the number of GPUs needed
                            .addEvaluator(new Accuracy()) // Model Accuracy
                            .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

            Trainer trainer = model.newTrainer(config);
            updater = (batchSize, subManager) -> trainer.step();
        }

        Function<String, String> predict =
                (prefix) -> predictCh8(prefix, 50, net, vocab, device, manager);
        // Train and predict
        double ppl = 0.0;
        double speed = 0.0;
        for (int epoch = 0; epoch < numEpochs; epoch++) {
            Pair<Double, Double> pair =
                    trainEpochCh8(net, dataset, loss, updater, device, useRandomIter, manager);
            ppl = pair.getKey();
            speed = pair.getValue();
            if ((epoch + 1) % 10 == 0) {
                animator.add(epoch + 1, (float) ppl, "ppl");
                animator.show();
            }
        }
        System.out.format(
                "perplexity: %.1f, %.1f tokens/sec on %s%n", ppl, speed, device.toString());
        System.out.println(predict.apply("time traveller"));
        System.out.println(predict.apply("traveller"));
    }

    /** Train a model within one epoch. */
    public static Pair<Double, Double> trainEpochCh8(
            Object net,
            RandomAccessDataset dataset,
            Loss loss,
            Functions.voidTwoFunction<Integer, NDManager> updater,
            Device device,
            boolean useRandomIter,
            NDManager manager)
            throws IOException, TranslateException {
        StopWatch watch = new StopWatch();
        watch.start();
        Accumulator metric = new Accumulator(2); // Sum of training loss, no. of tokens

        try (NDManager childManager = manager.newSubManager()) {
            NDList state = null;
            for (Batch batch : dataset.getData(manager)) {
                NDArray X = batch.getData().head().toDevice(Functions.tryGpu(0), true);
                X.attach(childManager);
                NDArray Y = batch.getLabels().head().toDevice(Functions.tryGpu(0), true);
                Y.attach(childManager);
                if (state == null || useRandomIter) {
                    // Initialize `state` when either it is the first iteration or
                    // using random sampling
                    if (net instanceof RNNModelScratch) {
                        state =
                                ((RNNModelScratch) net)
                                        .beginState((int) X.getShape().getShape()[0], device);
                    }
                } else {
                    for (NDArray s : state) {
                        s.stopGradient();
                    }
                }
                if (state != null) {
                    state.attach(childManager);
                }

                NDArray y = Y.transpose().reshape(new Shape(-1));
                X = X.toDevice(device, false);
                y = y.toDevice(device, false);
                try (GradientCollector gc = Engine.getInstance().newGradientCollector()) {
                    NDArray yHat;
                    if (net instanceof RNNModelScratch) {
                        Pair<NDArray, NDList> pairResult =
                                ((RNNModelScratch) net).forward(X, state);
                        yHat = pairResult.getKey();
                        state = pairResult.getValue();
                    } else {
                        NDList pairResult;
                        if (state == null) {
                            // Begin state
                            pairResult =
                                    ((AbstractBlock) net)
                                            .forward(
                                                    new ParameterStore(manager, false),
                                                    new NDList(X),
                                                    true);
                        } else {
                            pairResult =
                                    ((AbstractBlock) net)
                                            .forward(
                                                    new ParameterStore(manager, false),
                                                    new NDList(X).addAll(state),
                                                    true);
                        }
                        yHat = pairResult.get(0);
                        state = pairResult.subNDList(1);
                    }

                    NDArray l = loss.evaluate(new NDList(y), new NDList(yHat)).mean();
                    gc.backward(l);
                    metric.add(new float[] {l.getFloat() * y.size(), y.size()});
                }
                gradClipping(net, 1, childManager);
                updater.apply(1, childManager); // Since the `mean` function has been invoked
            }
        }
        return new Pair<>(Math.exp(metric.get(0) / metric.get(1)), metric.get(1) / watch.stop());
    }

    /** Clip the gradient. */
    public static void gradClipping(Object net, int theta, NDManager manager) {
        double result = 0;
        NDList params;
        if (net instanceof RNNModelScratch) {
            params = ((RNNModelScratch) net).params;
        } else {
            params = new NDList();
            for (Pair<String, Parameter> pair : ((AbstractBlock) net).getParameters()) {
                params.add(pair.getValue().getArray());
            }
        }
        for (NDArray p : params) {
            NDArray gradient = p.getGradient().stopGradient();
            gradient.attach(manager);
            result += gradient.pow(2).sum().getFloat();
        }
        double norm = Math.sqrt(result);
        if (norm > theta) {
            for (NDArray param : params) {
                NDArray gradient = param.getGradient();
                gradient.muli(theta / norm);
            }
        }
    }
}
