import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.norm.Dropout;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class PositionalEncoding extends AbstractBlock {

    private Dropout dropout;
    public NDArray P;

    public PositionalEncoding(int numHiddens, float dropout, int maxLen, NDManager manager) {
        this.dropout = Dropout.builder().optRate(dropout).build();
        this.addChildBlock("dropout", this.dropout);

        // Create a long enough `P`
        P = manager.zeros(new Shape(1, maxLen, numHiddens));
        NDArray X =
                manager.arange(maxLen)
                        .reshape(-1, 1)
                        .div(
                                manager.create(10000)
                                        .pow(manager.arange(0, numHiddens, 2).div(numHiddens)));
        P.set(new NDIndex(":, :, {}::{}", 0, 2), X.sin());
        P.set(new NDIndex(":, :, {}::{}", 1, 2), X.cos());
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray X = inputs.get(0);
        X = X.add(P.get(":, :{}, :", X.getShape().get(1)));
        return new NDList(dropout.forward(parameterStore, new NDList(X), training, params).get(0));
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {}
}
