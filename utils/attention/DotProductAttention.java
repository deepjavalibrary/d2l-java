import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/* Scaled dot product attention. */
public class DotProductAttention extends AbstractBlock {

    private Dropout dropout;
    public NDArray attentionWeights;
    private Shape[] outputShapes;

    public DotProductAttention(float dropout) {
        this.dropout = Dropout.builder().optRate(dropout).build();
        this.addChildBlock("dropout", this.dropout);
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore ps, NDList inputs, boolean training, PairList<String, Object> params) {
        // Shape of `queries`: (`batchSize`, no. of queries, `d`)
        // Shape of `keys`: (`batchSize`, no. of key-value pairs, `d`)
        // Shape of `values`: (`batchSize`, no. of key-value pairs, value
        // dimension)
        // Shape of `valid_lens`: (`batchSize`,) or (`batchSize`, no. of queries)
        NDArray queries = inputs.get(0);
        NDArray keys = inputs.get(1);
        NDArray values = inputs.get(2);
        NDArray validLens = inputs.get(3);

        // Swap the last two dimensions of `keys` and perform batchDot
        NDArray scores = queries.batchDot(keys.swapAxes(1, 2)).div(Math.sqrt(2));
        attentionWeights = Chap10Utils.maskedSoftmax(scores, validLens);
        NDList result = dropout.forward(ps, new NDList(attentionWeights), training, params);
        return new NDList(result.get(0).batchDot(values));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return outputShapes;
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        try (NDManager sub = manager.newSubManager()) {
            NDArray queries = sub.zeros(inputShapes[0], dataType);
            NDArray keys = sub.zeros(inputShapes[1], dataType);
            NDArray scores = queries.batchDot(keys.swapAxes(1, 2));
            Shape[] shapes = {scores.getShape()};
            dropout.initialize(manager, dataType, shapes);
            outputShapes = dropout.getOutputShapes(shapes);
        }
    }
}
