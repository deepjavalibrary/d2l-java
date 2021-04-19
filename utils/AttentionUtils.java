import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.UniformInitializer;
import ai.djl.util.PairList;

public class Chap10Utils {

    public static NDArray maskedSoftmax(NDArray X, NDArray validLens) {
        /* Perform softmax operation by masking elements on the last axis. */
        // `X`: 3D tensor, `validLens`: 1D or 2D tensor
        if (validLens == null) {
            return X.softmax(-1);
        } else {
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
}


/* Scaled dot product attention. */
public class DotProductAttention extends AbstractBlock {
    private static final byte VERSION = 1;
    private Dropout dropout;
    public NDArray attentionWeights;

    public DotProductAttention(float dropout) {
        super(VERSION);

        this.dropout = Dropout.builder().optRate(dropout).build();
        this.addChildBlock("dropout", this.dropout);
        this.dropout.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
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

        Long d = queries.getShape().get(queries.getShape().dimension() - 1);
        // Swap the last two dimensions of `keys` and perform batchDot
        NDArray scores = queries.batchDot(keys.swapAxes(1, 2)).div(Math.sqrt(2));
        attentionWeights = Chap10Utils.maskedSoftmax(scores, validLens);
        return new NDList(
                this.dropout
                        .forward(
                                parameterStore, new NDList(this.attentionWeights), training, params)
                        .get(0)
                        .batchDot(values));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {}
}

public class MultiHeadAttention extends AbstractBlock {
    private static final byte VERSION = 1;
    private int numHeads;
    public DotProductAttention attention;
    private Linear W_k;
    private Linear W_q;
    private Linear W_v;
    private Linear W_o;
    private Dropout dropout;

    public MultiHeadAttention(int numHiddens, int numHeads, float dropout, boolean useBias) {
        super(VERSION);
        this.numHeads = numHeads;

        attention = new DotProductAttention(dropout);

        this.W_q = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
        this.W_q.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
        this.addChildBlock("W_q", this.W_q);

        this.W_k = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
        this.W_k.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
        this.addChildBlock("W_k", this.W_k);

        this.W_v = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
        this.W_v.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
        this.addChildBlock("W_v", this.W_v);

        this.W_o = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
        this.W_o.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
        this.addChildBlock("W_o", this.W_o);

        this.dropout = Dropout.builder().optRate(dropout).build();
        this.addChildBlock("dropout", this.dropout);
        this.dropout.setInitializer(new UniformInitializer(0.07f), Parameter.Type.WEIGHT);
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
                Chap10Utils.transposeQkv(
                        W_q.forward(parameterStore, new NDList(queries), training, params).get(0),
                        this.numHeads);
        keys =
                Chap10Utils.transposeQkv(
                        W_k.forward(parameterStore, new NDList(keys), training, params).get(0),
                        this.numHeads);
        values =
                Chap10Utils.transposeQkv(
                        W_v.forward(parameterStore, new NDList(values), training, params).get(0),
                        this.numHeads);

        if (validLens != null) {
            // On axis 0, copy the first item (scalar or vector) for
            // `numHeads` times, then copy the next item, and so on
            validLens = validLens.repeat(0, this.numHeads);
        }

        // Shape of `output`: (`batchSize` * `numHeads`, no. of queries,
        // `numHiddens` / `numHeads`)
        NDArray output =
                this.attention
                        .forward(
                                parameterStore,
                                new NDList(queries, keys, values, validLens),
                                training,
                                params)
                        .get(0);

        // Shape of `outputConcat`:
        // (`batchSize`, no. of queries, `numHiddens`)
        NDArray outputConcat = Chap10Utils.transposeOutput(output, this.numHeads);
        return new NDList(
                this.W_o.forward(parameterStore, new NDList(outputConcat), training, params)
                        .get(0));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {}
}