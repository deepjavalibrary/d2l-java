import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.util.Pair;

/** An RNN Model implemented from scratch. */
public class RNNModelScratch {

    public int vocabSize;
    public int numHiddens;
    public NDList params;
    public Functions.TriFunction<Integer, Integer, Device, NDList> initState;
    public Functions.TriFunction<NDArray, NDList, NDList, Pair<NDArray, NDList>> forwardFn;

    public RNNModelScratch(
            int vocabSize,
            int numHiddens,
            Device device,
            Functions.TriFunction<Integer, Integer, Device, NDList> getParams,
            Functions.TriFunction<Integer, Integer, Device, NDList> initRNNState,
            Functions.TriFunction<NDArray, NDList, NDList, Pair<NDArray, NDList>> forwardFn) {
        this.vocabSize = vocabSize;
        this.numHiddens = numHiddens;
        this.params = getParams.apply(vocabSize, numHiddens, device);
        this.initState = initRNNState;
        this.forwardFn = forwardFn;
    }

    public Pair<NDArray, NDList> forward(NDArray X, NDList state) {
        X = X.transpose().oneHot(vocabSize);
        return forwardFn.apply(X, state, params);
    }

    public NDList beginState(int batchSize, Device device) {
        return initState.apply(batchSize, numHiddens, device);
    }
}
