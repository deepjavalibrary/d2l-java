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

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.UniformInitializer;
import ai.djl.util.PairList;

public class JupyterAttentionScoringFunctions {

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();

        maskedSoftmax(
                manager.randomUniform(0, 1, new Shape(2, 2, 4)),
                manager.create(new float[] {2, 3}));

        maskedSoftmax(
                manager.randomUniform(0, 1, new Shape(2, 2, 4)),
                manager.create(new float[][] {{1, 3}, {2, 4}}));

        NDArray queries = manager.randomNormal(0, 1, new Shape(2, 1, 20), DataType.FLOAT32);
        NDArray keys = manager.ones(new Shape(2, 10, 2));
        // The two value matrices in the `values` minibatch are identical
        NDArray values = manager.arange(40f).reshape(1, 10, 4).repeat(0, 2);
        NDArray validLens = manager.create(new float[] {2, 6});

        AdditiveAttention attention = new AdditiveAttention(8, 0.1f);
        NDList input = new NDList(queries, keys, values, validLens);
        ParameterStore ps = new ParameterStore(manager, false);
        attention.initialize(manager, DataType.FLOAT32, input.getShapes());
        attention.forward(ps, input, false).head();

        PlotUtils.showHeatmaps(
                attention.attentionWeights.reshape(1, 1, 2, 10),
                "Keys",
                "Queries",
                new String[] {""},
                500,
                700);

        queries = manager.randomNormal(0, 1, new Shape(2, 1, 2), DataType.FLOAT32);
        DotProductAttention productAttention = new DotProductAttention(0.5f);
        input = new NDList(queries, keys, values, validLens);
        attention.initialize(manager, DataType.FLOAT32, input.getShapes());
        productAttention.forward(ps, input, false).head();
    }

    public static NDArray maskedSoftmax(NDArray X, NDArray validLens) {
        /* Perform softmax operation by masking elements on the last axis. */
        // `X`: 3D NDArray, `validLens`: 1D or 2D NDArray
        if (validLens == null) {
            return X.softmax(-1);
        }

        Shape shape = X.getShape();
        if (validLens.getShape().dimension() == 1) {
            validLens = validLens.repeat(shape.get(1));
        } else {
            validLens = validLens.reshape(-1);
        }
        // On the last axis, replace masked elements with a very large negative
        // value, whose exponentiation outputs 0
        X =
                X.reshape(new Shape(-1, shape.get(shape.dimension() - 1)))
                        .sequenceMask(validLens, (float) -1E6);
        return X.softmax(-1).reshape(shape);
    }

    /* Additive attention. */
    public static class AdditiveAttention extends AbstractBlock {

        private Linear W_k;
        private Linear W_q;
        private Linear W_v;
        private Dropout dropout;
        public NDArray attentionWeights;

        public AdditiveAttention(int numHiddens, float dropout) {
            W_k = Linear.builder().setUnits(numHiddens).optBias(false).build();
            addChildBlock("W_k", W_k);

            W_q = Linear.builder().setUnits(numHiddens).optBias(false).build();
            addChildBlock("W_q", W_q);

            W_v = Linear.builder().setUnits(1).optBias(false).build();
            addChildBlock("W_v", W_v);

            this.dropout = Dropout.builder().optRate(dropout).build();
            addChildBlock("dropout", this.dropout);
        }

        @Override
        protected NDList forwardInternal(
                ParameterStore ps,
                NDList inputs,
                boolean training,
                PairList<String, Object> params) {
            // Shape of the output `queries` and `attentionWeights`:
            // (no. of queries, no. of key-value pairs)
            NDArray queries = inputs.get(0);
            NDArray keys = inputs.get(1);
            NDArray values = inputs.get(2);
            NDArray validLens = inputs.get(3);

            queries = W_q.forward(ps, new NDList(queries), training, params).head();
            keys = W_k.forward(ps, new NDList(keys), training, params).head();
            // After dimension expansion, shape of `queries`: (`batchSize`, no. of
            // queries, 1, `numHiddens`) and shape of `keys`: (`batchSize`, 1,
            // no. of key-value pairs, `numHiddens`). Sum them up with
            // broadcasting
            NDArray features = queries.expandDims(2).add(keys.expandDims(1));
            features = features.tanh();
            // There is only one output of `this.W_v`, so we remove the last
            // one-dimensional entry from the shape. Shape of `scores`:
            // (`batchSize`, no. of queries, no. of key-value pairs)
            NDArray result = W_v.forward(ps, new NDList(features), training, params).head();
            NDArray scores = result.squeeze(-1);
            attentionWeights = maskedSoftmax(scores, validLens);
            // Shape of `values`: (`batchSize`, no. of key-value pairs, value dimension)
            NDList list = dropout.forward(ps, new NDList(attentionWeights), training, params);
            return new NDList(list.head().batchDot(values));
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            throw new UnsupportedOperationException("Not implemented");
        }

        @Override
        public void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {
            W_q.initialize(manager, dataType, inputShapes[0]);
            W_k.initialize(manager, dataType, inputShapes[1]);
            long[] q = W_q.getOutputShapes(new Shape[] {inputShapes[0]})[0].getShape();
            long[] k = W_k.getOutputShapes(new Shape[] {inputShapes[1]})[0].getShape();
            long w = Math.max(q[q.length - 2], k[k.length - 2]);
            long h = Math.max(q[q.length - 1], k[k.length - 1]);
            long[] shape = new long[] {2, 1, w, h};
            W_v.initialize(manager, dataType, new Shape(shape));
            long[] dropoutShape = new long[shape.length - 1];
            System.arraycopy(shape, 0, dropoutShape, 0, dropoutShape.length);
            dropout.initialize(manager, dataType, new Shape(dropoutShape));
        }
    }

    /* Scaled dot product attention. */
    public static final class DotProductAttention extends AbstractBlock {

        private Dropout dropout;
        public NDArray attentionWeights;

        public DotProductAttention(float dropout) {
            this.dropout = Dropout.builder().optRate(dropout).build();
            this.addChildBlock("dropout", this.dropout);
            this.dropout.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
        }

        @Override
        protected NDList forwardInternal(
                ParameterStore ps,
                NDList inputs,
                boolean training,
                PairList<String, Object> params) {
            // Shape of `queries`: (`batchSize`, no. of queries, `d`)
            // Shape of `keys`: (`batchSize`, no. of key-value pairs, `d`)
            // Shape of `values`: (`batchSize`, no. of key-value pairs, value
            // dimension)
            // Shape of `valid_lens`: (`batchSize`,) or (`batchSize`, no. of queries)
            NDArray queries = inputs.get(0);
            NDArray keys = inputs.get(1);
            NDArray values = inputs.get(2);
            NDArray validLens = inputs.get(3);

            // Long d = queries.getShape().get(queries.getShape().dimension() - 1);
            // Swap the last two dimensions of `keys` and perform batchDot
            NDArray scores = queries.batchDot(keys.swapAxes(1, 2)).div(Math.sqrt(2));
            attentionWeights = maskedSoftmax(scores, validLens);
            NDList list = dropout.forward(ps, new NDList(attentionWeights), training, params);
            return new NDList(list.head().batchDot(values));
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            throw new UnsupportedOperationException("Not implemented");
        }

        @Override
        public void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {
            try (NDManager sub = manager.newSubManager()) {
                NDArray queries = sub.zeros(inputShapes[0], dataType);
                NDArray keys = sub.zeros(inputShapes[1], dataType);
                NDArray scores = queries.batchDot(keys.swapAxes(1, 2));
                dropout.initialize(manager, dataType, scores.getShape());
            }
        }
    }
}
