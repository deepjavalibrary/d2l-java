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

public class MultiHeadAttention extends AbstractBlock {

    private static final byte VERSION = 1;

    private int numHeads;
    public DotProductAttention attention;
    private Linear W_k;
    private Linear W_q;
    private Linear W_v;
    private Linear W_o;

    public MultiHeadAttention(int numHiddens, int numHeads, float dropout, boolean useBias) {
        super(VERSION);
        this.numHeads = numHeads;

        attention = new DotProductAttention(dropout);

        this.W_q = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
        this.addChildBlock("W_q", this.W_q);

        this.W_k = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
        this.addChildBlock("W_k", this.W_k);

        this.W_v = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
        this.addChildBlock("W_v", this.W_v);

        this.W_o = Linear.builder().setUnits(numHiddens).optBias(useBias).build();
        this.addChildBlock("W_o", this.W_o);

        Dropout dropout1 = Dropout.builder().optRate(dropout).build();
        this.addChildBlock("dropout", dropout1);
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
