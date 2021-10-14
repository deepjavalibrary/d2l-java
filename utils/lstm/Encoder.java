import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;

/** The base encoder interface for the encoder-decoder architecture. */
public abstract class Encoder extends AbstractBlock {

    @Override
    public Shape[] getOutputShapes(Shape[] inputShapes) {
        throw new UnsupportedOperationException("Not implemented");
    }
}
