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
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class JupyterMultiHeadAttention {

    public static void main(String[] args) {
        NDManager manager = NDManager.newBaseManager();

        int numHiddens = 100;
        int numHeads = 5;
        MultiHeadAttention attention = new MultiHeadAttention(numHiddens, numHeads, 0.5f, false);

        int batchSize = 2;
        int numQueries = 4;
        int numKvpairs = 6;
        NDArray validLens = manager.create(new float[] {3, 2});
        NDArray X = manager.ones(new Shape(batchSize, numQueries, numHiddens));
        NDArray Y = manager.ones(new Shape(batchSize, numKvpairs, numHiddens));

        ParameterStore ps = new ParameterStore(manager, false);
        NDList result = attention.forward(ps, new NDList(X, Y, Y, validLens), true);
        result.get(0).getShape();
    }

    public static NDArray transposeQkv(NDArray X, int numHeads) {
        // Shape of input `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHiddens`).
        // Shape of output `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHeads`,
        // `numHiddens` / `numHeads`)
        X = X.reshape(X.getShape().get(0), X.getShape().get(1), numHeads, -1);

        // Shape of output `X`:
        // (`batchSize`, `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        X = X.transpose(0, 2, 1, 3);

        // Shape of `output`:
        // (`batchSize` * `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        return X.reshape(-1, X.getShape().get(2), X.getShape().get(3));
    }

    public static NDArray transposeOutput(NDArray X, int numHeads) {
        X = X.reshape(-1, numHeads, X.getShape().get(1), X.getShape().get(2));
        X = X.transpose(0, 2, 1, 3);
        return X.reshape(X.getShape().get(0), X.getShape().get(1), -1);
    }

    public static class MultiHeadAttention extends AbstractBlock {

        private int numHeads;
        public DotProductAttention attention;
        private Linear W_k;
        private Linear W_q;
        private Linear W_v;
        private Linear W_o;

        public MultiHeadAttention(int numHiddens, int numHeads, float dropout, boolean useBias) {
            this.numHeads = numHeads;

            attention = new DotProductAttention(dropout);

            W_q = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
            addChildBlock("W_q", W_q);

            W_k = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
            addChildBlock("W_k", W_k);

            W_v = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
            addChildBlock("W_v", W_v);

            W_o = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
            addChildBlock("W_o", W_o);

            Dropout dropout1 = Dropout.builder().optRate(dropout).build();
            addChildBlock("dropout", dropout1);
        }

        @Override
        protected NDList forwardInternal(
                ParameterStore parameterStore,
                NDList inputs,
                boolean training,
                PairList<String, Object> params) {
            // Shape of `queries`, `keys`, or `values`:
            // (`batchSize`, no. of queries or key-value pairs, `numHiddens`)
            // Shape of `validLens`:
            // (`batchSize`,) or (`batchSize`, no. of queries)
            // After transposing, shape of output `queries`, `keys`, or `values`:
            // (`batchSize` * `numHeads`, no. of queries or key-value pairs,
            // `numHiddens` / `numHeads`)
            NDArray queries = inputs.get(0);
            NDArray keys = inputs.get(1);
            NDArray values = inputs.get(2);
            NDArray validLens = inputs.get(3);

            queries =
                    transposeQkv(
                            W_q.forward(parameterStore, new NDList(queries), training, params)
                                    .get(0),
                            numHeads);
            keys =
                    transposeQkv(
                            W_k.forward(parameterStore, new NDList(keys), training, params).get(0),
                            numHeads);
            values =
                    transposeQkv(
                            W_v.forward(parameterStore, new NDList(values), training, params)
                                    .get(0),
                            numHeads);

            if (validLens != null) {
                // On axis 0, copy the first item (scalar or vector) for
                // `numHeads` times, then copy the next item, and so on
                validLens = validLens.repeat(0, numHeads);
            }

            // Shape of `output`: (`batchSize` * `numHeads`, no. of queries,
            // `numHiddens` / `numHeads`)
            NDArray output =
                    attention
                            .forward(
                                    parameterStore,
                                    new NDList(queries, keys, values, validLens),
                                    training,
                                    params)
                            .get(0);

            // Shape of `outputConcat`:
            // (`batchSize`, no. of queries, `numHiddens`)
            NDArray outputConcat = transposeOutput(output, numHeads);
            return new NDList(
                    W_o.forward(parameterStore, new NDList(outputConcat), training, params).get(0));
        }

        @Override
        public Shape[] getOutputShapes(Shape[] inputShapes) {
            throw new UnsupportedOperationException("Not implemented");
        }

        @Override
        public void initializeChildBlocks(
                NDManager manager, DataType dataType, Shape... inputShapes) {}
    }
}
