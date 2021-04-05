import ai.djl.ndarray.*;
import ai.djl.ndarray.types.Shape;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.AbstractBlock;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;

public abstract class Encoder extends AbstractBlock {
    /* The base encoder interface for the encoder-decoder architecture. */
    private static final byte VERSION = 1;

    public Encoder() {
        super(VERSION);
    }

    @Override
    abstract protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params);

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}

public abstract class Decoder extends AbstractBlock {
    /* The base decoder interface for the encoder-decoder architecture. */
    private static final byte VERSION = 1;
    public NDArray attentionWeights;

    public Decoder() {
        super(VERSION);
    }

    @Override
    abstract protected NDList forwardInternal(
            ParameterStore parameterStore,
            NDList inputs,
            boolean training,
            PairList<String, Object> params);

    abstract public NDList beginState(NDList encOutputs);

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}

public class EncoderDecoder extends AbstractBlock {
    /* The base class for the encoder-decoder architecture. */
    private static final byte VERSION = 1;
    public Encoder encoder;
    public Decoder decoder;

    public EncoderDecoder(Encoder encoder, Decoder decoder) {
        super(VERSION);

        this.encoder = encoder;
        this.addChildBlock("encoder", this.encoder);
        this.decoder = decoder;
        this.addChildBlock("decoder", this.decoder);
    }

    /** {@inheritDoc} */
    @Override
    public void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    }

    @Override
    protected NDList forwardInternal(ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
        NDArray encX = inputs.get(0);
        NDArray decX = inputs.get(1);
        NDList encOutputs = this.encoder.forward(parameterStore, new NDList(encX), training, params);
        NDList decState = this.decoder.beginState(encOutputs);
        return this.decoder.forward(parameterStore, new NDList(decX).addAll(decState), training, params);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
