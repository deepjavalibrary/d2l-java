import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public class RNNModel<T extends AbstractBlock> extends AbstractBlock {

    private T rnnLayer;
    private Linear dense;
    private int vocabSize;

    public RNNModel(T rnnLayer, int vocabSize) {
        this.rnnLayer = rnnLayer;
        this.addChildBlock("rnn", rnnLayer);
        this.vocabSize = vocabSize;
        this.dense = Linear.builder().setUnits(vocabSize).build();
        this.addChildBlock("linear", dense);
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
        /* rnnLayer is already initialized so we don't have to do anything here, just override it.*/
    }

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray X = inputs.get(0).transpose().oneHot(vocabSize);
        inputs.set(0, X);
        NDList result = rnnLayer.forward(parameterStore, inputs, training);
        NDArray Y = result.get(0);
        NDList state = result.subNDList(1);

        int shapeLength = Y.getShape().getShape().length;
        NDList output =
                dense.forward(
                        parameterStore,
                        new NDList(Y.reshape(new Shape(-1, Y.getShape().get(shapeLength - 1)))),
                        training);
        return new NDList(output.get(0)).addAll(state);
    }

    /* We won't implement this since we won't be using it but it's required as part of an AbstractBlock  */
    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        return new Shape[0];
    }
}
