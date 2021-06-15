import ai.djl.ndarray.*;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;

public class TrainingChapter9 {

    /** Clip the gradient. */
    public static void gradClipping(Object net, int theta, NDManager manager) {
        double result = 0;
        NDList params;
        params = new NDList();
        for (Pair<String, Parameter> pair : ((AbstractBlock) net).getParameters()) {
            params.add(pair.getValue().getArray());
        }
        for (NDArray p : params) {
            NDArray gradient = p.getGradient().stopGradient();
            gradient.attach(manager);
            result += gradient.pow(2).sum().getFloat();
        }
        double norm = Math.sqrt(result);
        if (norm > theta) {
            for (NDArray param : params) {
                NDArray gradient = param.getGradient();
                gradient.muli(theta / norm);
            }
        }
    }
}
