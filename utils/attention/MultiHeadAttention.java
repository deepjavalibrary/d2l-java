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
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
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
        // On axis 0, copy the first item (scalar or vector) for
        // `numHeads` times, then copy the next item, and so on
        validLens = validLens.repeat(0, numHeads);

        queries =
                Chap10Utils.transposeQkv(
                        W_q.forward(ps, new NDList(queries), training, params).get(0), numHeads);
        keys =
                Chap10Utils.transposeQkv(
                        W_k.forward(ps, new NDList(keys), training, params).get(0), numHeads);
        values =
                Chap10Utils.transposeQkv(
                        W_v.forward(ps, new NDList(values), training, params).get(0), numHeads);

        // Shape of `output`: (`batchSize` * `numHeads`, no. of queries,
        // `numHiddens` / `numHeads`)
        NDArray output =
                attention
                        .forward(ps, new NDList(queries, keys, values, validLens), training, params)
                        .get(0);

        // Shape of `outputConcat`:
        // (`batchSize`, no. of queries, `numHiddens`)
        NDArray outputConcat = Chap10Utils.transposeOutput(output, numHeads);
        return new NDList(W_o.forward(ps, new NDList(outputConcat), training, params).get(0));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        try (NDManager sub = manager.newSubManager()) {
            NDArray queries = sub.zeros(inputShapes[0], dataType);
            NDArray keys = sub.zeros(inputShapes[1], dataType);
            NDArray values = sub.zeros(inputShapes[2], dataType);
            NDArray validLens = sub.zeros(inputShapes[3], dataType);
            validLens = validLens.repeat(0, numHeads);

            ParameterStore ps = new ParameterStore(sub, false);

            W_q.initialize(manager, dataType, queries.getShape());
            W_k.initialize(manager, dataType, keys.getShape());
            W_v.initialize(manager, dataType, values.getShape());

            queries =
                    Chap10Utils.transposeQkv(
                            W_q.forward(ps, new NDList(queries), false).get(0), numHeads);
            keys =
                    Chap10Utils.transposeQkv(
                            W_k.forward(ps, new NDList(keys), false).get(0), numHeads);
            values =
                    Chap10Utils.transposeQkv(
                            W_v.forward(ps, new NDList(values), false).get(0), numHeads);

            NDList list = new NDList(queries, keys, values, validLens);
            attention.initialize(sub, dataType, list.getShapes());
            NDArray output = attention.forward(ps, list, false).head();
            NDArray outputConcat = Chap10Utils.transposeOutput(output, numHeads);

            W_o.initialize(manager, dataType, outputConcat.getShape());
        }
    }
}
