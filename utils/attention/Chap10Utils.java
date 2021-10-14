import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.Shape;

public class Chap10Utils {

    public static NDArray maskedSoftmax(NDArray X, NDArray validLens) {
        /* Perform softmax operation by masking elements on the last axis. */
        // `X`: 3D tensor, `validLens`: 1D or 2D tensor
        if (validLens == null) {
            return X.softmax(-1);
        }

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

    public static NDArray transposeQkv(NDArray X, int numHeads) {
        // Shape of input `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHiddens`).
        // Shape of output `X`:
        // (`batchSize`, no. of queries or key-value pairs, `numHeads`,
        // `numHiddens` / `numHeads`)
        X = X.reshape(X.getShape().get(0), X.getShape().get(1), numHeads, -1);

        // Shape of output `X`:
        // (`batchSize`, `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        X = X.transpose(0, 2, 1, 3);

        // Shape of `output`:
        // (`batchSize` * `numHeads`, no. of queries or key-value pairs,
        // `numHiddens` / `numHeads`)
        return X.reshape(-1, X.getShape().get(2), X.getShape().get(3));
    }

    public static NDArray transposeOutput(NDArray X, int numHeads) {
        X = X.reshape(-1, numHeads, X.getShape().get(1), X.getShape().get(2));
        X = X.transpose(0, 2, 1, 3);
        return X.reshape(X.getShape().get(0), X.getShape().get(1), -1);
    }
}
