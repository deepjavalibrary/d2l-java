import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;

/** The base encoder interface for the encoder-decoder architecture. */
public abstract class Encoder extends AbstractBlock {

    private static final byte VERSION = 1;

    public Encoder() {
        super(VERSION);
    }

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
