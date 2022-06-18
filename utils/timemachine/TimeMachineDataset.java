import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.training.dataset.Record;
import ai.djl.translate.TranslateException;
import ai.djl.util.Pair;
import ai.djl.util.Progress;

import java.io.IOException;
import java.util.List;
import java.util.Random;

public class TimeMachineDataset extends RandomAccessDataset {

    private Vocab vocab;
    private NDArray data;
    private NDArray labels;
    private int numSteps;
    private int maxTokens;
    private int batchSize;
    private NDManager manager;
    private boolean prepared;

    public TimeMachineDataset(Builder builder) {
        super(builder);
        this.numSteps = builder.numSteps;
        this.maxTokens = builder.maxTokens;
        this.batchSize = builder.getSampler().getBatchSize();
        this.manager = builder.manager;
        this.data = this.manager.create(new Shape(0, 35), DataType.INT32);
        this.labels = this.manager.create(new Shape(0, 35), DataType.INT32);
        this.prepared = false;
    }

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        NDArray X = data.get(new NDIndex("{}", index));
        NDArray Y = labels.get(new NDIndex("{}", index));
        return new Record(new NDList(X), new NDList(Y));
    }

    @Override
    protected long availableSize() {
        return data.getShape().get(0);
    }

    @Override
    public void prepare(Progress progress) throws IOException {
        if (prepared) {
            return;
        }

        Pair<List<Integer>, Vocab> corpusVocabPair = TimeMachine.loadCorpusTimeMachine(maxTokens);
        List<Integer> corpus = corpusVocabPair.getKey();
        this.vocab = corpusVocabPair.getValue();

        // Start with a random offset (inclusive of `numSteps - 1`) to partition a
        // sequence
        int offset = new Random().nextInt(numSteps);
        int numTokens = (corpus.size() - offset - 1) / batchSize * batchSize;
        NDArray Xs =
                manager.create(
                        corpus.subList(offset, offset + numTokens).stream()
                                .mapToInt(Integer::intValue)
                                .toArray());
        NDArray Ys =
                manager.create(
                        corpus.subList(offset + 1, offset + 1 + numTokens).stream()
                                .mapToInt(Integer::intValue)
                                .toArray());
        Xs = Xs.reshape(new Shape(batchSize, -1));
        Ys = Ys.reshape(new Shape(batchSize, -1));
        int numBatches = (int) Xs.getShape().get(1) / numSteps;

        for (int i = 0; i < numSteps * numBatches; i += numSteps) {
            NDArray X = Xs.get(new NDIndex(":, {}:{}", i, i + numSteps));
            NDArray Y = Ys.get(new NDIndex(":, {}:{}", i, i + numSteps));
            // Temp variables to be able to detach NDArray which will be replaced
            NDArray temp = this.data;
            NDArray temp2 = this.data;
            this.data = this.data.concat(X);
            this.labels = this.labels.concat(Y);
            temp.detach();
            temp2.detach();
        }
        this.prepared = true;
    }

    public Vocab getVocab() {
        return this.vocab;
    }

    public static final class Builder extends BaseBuilder<Builder> {

        int numSteps;
        int maxTokens;
        NDManager manager;

        @Override
        protected Builder self() {
            return this;
        }

        public Builder setSteps(int steps) {
            this.numSteps = steps;
            return this;
        }

        public Builder setMaxTokens(int maxTokens) {
            this.maxTokens = maxTokens;
            return this;
        }

        public Builder setManager(NDManager manager) {
            this.manager = manager;
            return this;
        }

        public TimeMachineDataset build() throws IOException, TranslateException {
            return new TimeMachineDataset(this);
        }
    }
}
