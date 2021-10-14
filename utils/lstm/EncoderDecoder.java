import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

/** The base class for the encoder-decoder architecture. */
public class EncoderDecoder extends AbstractBlock {

    protected Encoder encoder;
    protected Decoder decoder;

    public EncoderDecoder(Encoder encoder, Decoder decoder) {
        this.encoder = encoder;
        this.addChildBlock("encoder", this.encoder);
        this.decoder = decoder;
        this.addChildBlock("decoder", this.decoder);
    }

    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {}

    @Override
    protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params) {
        NDArray encX = inputs.get(0);
        NDArray decX = inputs.get(1);
        NDList encOutputs = encoder.forward(parameterStore, new NDList(encX), training, params);
        NDList decState = decoder.initState(encOutputs);
        return decoder.forward(parameterStore, new NDList(decX).addAll(decState), training, params);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
