import ai.djl.ndarray.*;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
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
