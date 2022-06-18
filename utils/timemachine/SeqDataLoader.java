import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.util.Pair;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.Random;

public class SeqDataLoader implements Iterable<NDList> {

    public List<NDList> dataIter;
    public List<Integer> corpus;
    public Vocab vocab;
    public int batchSize;
    public int numSteps;

    /* An iterator to load sequence data. */
    public SeqDataLoader(
            int batchSize, int numSteps, boolean useRandomIter, int maxTokens, NDManager manager)
            throws IOException {
        Pair<List<Integer>, Vocab> corpusVocabPair = TimeMachine.loadCorpusTimeMachine(maxTokens);
        this.corpus = corpusVocabPair.getKey();
        this.vocab = corpusVocabPair.getValue();

        this.batchSize = batchSize;
        this.numSteps = numSteps;
        if (useRandomIter) {
            dataIter = seqDataIterRandom(corpus, batchSize, numSteps, manager);
        } else {
            dataIter = seqDataIterSequential(corpus, batchSize, numSteps, manager);
        }
    }

    @Override
    public Iterator<NDList> iterator() {
        return dataIter.iterator();
    }

    /** Return the iterator and the vocabulary of the time machine dataset. */
    public static Pair<List<NDList>, Vocab> loadDataTimeMachine(
            int batchSize, int numSteps, boolean useRandomIter, int maxTokens, NDManager manager)
            throws IOException {
        SeqDataLoader seqData =
                new SeqDataLoader(batchSize, numSteps, useRandomIter, maxTokens, manager);
        return new Pair<>(seqData.dataIter, seqData.vocab); // ArrayList<NDList>, Vocab
    }

    /** Generate a minibatch of subsequences using random sampling. */
    public List<NDList> seqDataIterRandom(
            List<Integer> corpus, int batchSize, int numSteps, NDManager manager) {
        // Start with a random offset (inclusive of `numSteps - 1`) to partition a
        // sequence
        corpus = corpus.subList(new Random().nextInt(numSteps - 1), corpus.size());
        // Subtract 1 since we need to account for labels
        int numSubseqs = (corpus.size() - 1) / numSteps;
        // The starting indices for subsequences of length `numSteps`
        List<Integer> initialIndices = new ArrayList<>();
        for (int i = 0; i < numSubseqs * numSteps; i += numSteps) {
            initialIndices.add(i);
        }
        // In random sampling, the subsequences from two adjacent random
        // minibatches during iteration are not necessarily adjacent on the
        // original sequence
        Collections.shuffle(initialIndices);

        int numBatches = numSubseqs / batchSize;

        ArrayList<NDList> pairs = new ArrayList<>();
        for (int i = 0; i < batchSize * numBatches; i += batchSize) {
            // Here, `initialIndices` contains randomized starting indices for
            // subsequences
            List<Integer> initialIndicesPerBatch = initialIndices.subList(i, i + batchSize);

            NDArray xNDArray =
                    manager.create(
                            new Shape(initialIndicesPerBatch.size(), numSteps), DataType.FLOAT32);
            NDArray yNDArray =
                    manager.create(
                            new Shape(initialIndicesPerBatch.size(), numSteps), DataType.FLOAT32);
            for (int j = 0; j < initialIndicesPerBatch.size(); j++) {
                List<Integer> X = data(initialIndicesPerBatch.get(j), corpus, numSteps);
                xNDArray.set(
                        new NDIndex(j),
                        manager.create(X.stream().mapToInt(Integer::intValue).toArray()));
                List<Integer> Y = data(initialIndicesPerBatch.get(j) + 1, corpus, numSteps);
                yNDArray.set(
                        new NDIndex(j),
                        manager.create(Y.stream().mapToInt(Integer::intValue).toArray()));
            }
            NDList pair = new NDList();
            pair.add(xNDArray);
            pair.add(yNDArray);
            pairs.add(pair);
        }
        return pairs;
    }

    List<Integer> data(int pos, List<Integer> corpus, int numSteps) {
        // Return a sequence of length `numSteps` starting from `pos`
        return new ArrayList<>(corpus.subList(pos, pos + numSteps));
    }

    /** Generate a minibatch of subsequences using sequential partitioning. */
    public List<NDList> seqDataIterSequential(
            List<Integer> corpus, int batchSize, int numSteps, NDManager manager) {
        // Start with a random offset to partition a sequence
        int offset = new Random().nextInt(numSteps);
        int numTokens = ((corpus.size() - offset - 1) / batchSize) * batchSize;

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

        ArrayList<NDList> pairs = new ArrayList<>();
        for (int i = 0; i < numSteps * numBatches; i += numSteps) {
            NDArray X = Xs.get(new NDIndex(":, {}:{}", i, i + numSteps));
            NDArray Y = Ys.get(new NDIndex(":, {}:{}", i, i + numSteps));
            NDList pair = new NDList();
            pair.add(X);
            pair.add(Y);
            pairs.add(pair);
        }
        return pairs;
    }
}
