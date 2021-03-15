import ai.djl.ndarray.*;
import ai.djl.ndarray.types.*;
import ai.djl.ndarray.index.*;
import ai.djl.util.Pair;

import java.io.*;
import java.net.URL;
import java.util.*;

public class Vocab {
    public int unk;
    public List<Map.Entry<String, Integer>> tokenFreqs;
    public List<String> idxToToken;
    public HashMap<String, Integer> tokenToIdx;

    public Vocab(String[][] tokens, int minFreq, String[] reservedTokens) {
        // Sort according to frequencies
        LinkedHashMap<String, Integer> counter = countCorpus2D(tokens);
        this.tokenFreqs = new ArrayList<Map.Entry<String, Integer>>(counter.entrySet()); 
        Collections.sort(tokenFreqs, 
            new Comparator<Map.Entry<String, Integer>>() { 
                public int compare(Map.Entry<String, Integer> o1, Map.Entry<String, Integer> o2) { 
                    return (o2.getValue()).compareTo(o1.getValue()); 
                }
            });
        
        // The index for the unknown token is 0
        this.unk = 0;
        List<String> uniqTokens = new ArrayList<>();
        uniqTokens.add("<unk>");
        Collections.addAll(uniqTokens, reservedTokens);
        for (Map.Entry<String, Integer> entry : tokenFreqs) {
            if (entry.getValue() >= minFreq && !uniqTokens.contains(entry.getKey())) {
                uniqTokens.add(entry.getKey());
            }
        }
        
        this.idxToToken = new ArrayList<>();
        this.tokenToIdx = new HashMap<>();
        for (String token : uniqTokens) {
            this.idxToToken.add(token);
            this.tokenToIdx.put(token, this.idxToToken.size()-1);
        }
    }
    
    public int length() {
        return this.idxToToken.size();
    }
    
    public Integer[] getIdxs(String[] tokens) {
        List<Integer> idxs = new ArrayList<>();
        for (String token : tokens) {
            idxs.add(getIdx(token));
        }
        return idxs.toArray(new Integer[0]);
        
    }
    
    public Integer getIdx(String token) {
        return this.tokenToIdx.getOrDefault(token, this.unk);
    }
    
    
}

/** 
 * Count token frequencies.
 */
public LinkedHashMap<String, Integer> countCorpus(String[] tokens) {
    
    LinkedHashMap<String, Integer> counter = new LinkedHashMap<>();
    if (tokens.length != 0) {
        for (String token : tokens) {
            counter.put(token, counter.getOrDefault(token, 0)+1);
        }
    }
    return counter;
}

/**
 * Flatten a list of token lists into a list of tokens
 */
public LinkedHashMap<String, Integer> countCorpus2D(String[][] tokens) {
    List<String> allTokens = new ArrayList<String>();
    for (int i = 0; i < tokens.length; i++) {
        for (int j = 0; j < tokens[i].length; j++) {
             if (tokens[i][j] != "") {
                allTokens.add(tokens[i][j]);
             }
        }
    }
    return countCorpus(allTokens.toArray(new String[0]));
}

public class TimeMachine {
    /**
     * Split text lines into word or character tokens.
     */
    public static String[][] tokenize(String[] lines, String token) throws Exception {
        String[][] output = new String[lines.length][];
        if (token == "word") {
            for (int i = 0; i < output.length; i++) {
                output[i] = lines[i].split(" ");
            }
        }else if (token == "char") {
            for (int i = 0; i < output.length; i++) {
                output[i] = lines[i].split("");
            }
        }else {
            throw new Exception("ERROR: unknown token type: " + token);
        }
        return output; 
    }

    /**
     * Read `The Time Machine` dataset and return an array of the lines
     */
    public static String[] readTimeMachine() throws IOException {
        URL url = new URL("http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt");
        String[] lines;
        try (BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()))) {
            lines = in.lines().toArray(String[]::new);
        }

        for (int i = 0; i < lines.length; i++) {
            lines[i] = lines[i].replaceAll("[^A-Za-z]+", " ").strip().toLowerCase();
        }
        return lines;
    }

    /** 
     * Return token indices and the vocabulary of the time machine dataset.
     */
    public static Pair<List<Integer>, Vocab> loadCorpusTimeMachine(int maxTokens) throws IOException, Exception {
        String[] lines = readTimeMachine();
        String[][] tokens = tokenize(lines, "char");
        Vocab vocab = new Vocab(tokens, 0, new String[0]);
        // Since each text line in the time machine dataset is not necessarily a
        // sentence or a paragraph, flatten all the text lines into a single list
        List<Integer> corpus = new ArrayList<>();
        for (int i = 0; i < tokens.length; i++) {
            for (int j = 0; j < tokens[i].length; j++) {
                if (tokens[i][j] != "") {
                    corpus.add(vocab.getIdx(tokens[i][j]));
                }
            }
        }
        if (maxTokens > 0) {
            corpus = corpus.subList(0, maxTokens);
        }
        return new Pair(corpus, vocab);
    }
}

public class SeqDataLoader implements Iterable<NDList> {
    public ArrayList<NDList> dataIter;
    public List<Integer> corpus;
    public Vocab vocab;
    public int batchSize;
    public int numSteps;
    
    /* An iterator to load sequence data. */
    @SuppressWarnings("unchecked")
    public SeqDataLoader(int batchSize, int numSteps, boolean useRandomIter, int maxTokens) throws IOException, Exception {
        Pair<List<Integer>, Vocab> corpusVocabPair = TimeMachine.loadCorpusTimeMachine(maxTokens);
        this.corpus = corpusVocabPair.getKey();
        this.vocab = corpusVocabPair.getValue();
        
        this.batchSize = batchSize;
        this.numSteps = numSteps;
        if (useRandomIter) {
            dataIter = seqDataIterRandom(corpus, batchSize, numSteps, manager);
        }else {
            dataIter = seqDataIterSequential(corpus, batchSize, numSteps, manager);
        }
    }
    
    @Override
    public Iterator<NDList> iterator() {
        return dataIter.iterator();
    }
}
/**
 * Return the iterator and the vocabulary of the time machine dataset.
 */ 
public Pair<ArrayList<NDList>, Vocab> loadDataTimeMachine(int batchSize, int numSteps, boolean useRandomIter, int maxTokens) throws IOException, Exception {
    
    SeqDataLoader seqData = new SeqDataLoader(batchSize, numSteps, useRandomIter, maxTokens);
    return new Pair(seqData.dataIter, seqData.vocab); // ArrayList<NDList>, Vocab
}

/**
 * Generate a minibatch of subsequences using random sampling.
 */
public ArrayList<NDList>
        seqDataIterRandom(List<Integer> corpus, int batchSize, int numSteps, NDManager manager) {
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
        
    ArrayList<NDList> pairs = new ArrayList<NDList>();
    for (int i = 0; i < batchSize * numBatches; i += batchSize) {
        // Here, `initialIndices` contains randomized starting indices for
        // subsequences
        List<Integer> initialIndicesPerBatch = initialIndices.subList(i, i + batchSize);

        NDArray xNDArray = manager.create(new Shape(initialIndicesPerBatch.size(), numSteps), DataType.INT32);
        NDArray yNDArray = manager.create(new Shape(initialIndicesPerBatch.size(), numSteps), DataType.INT32);
        for (int j = 0; j < initialIndicesPerBatch.size(); j++) {
            ArrayList<Integer> X = data(initialIndicesPerBatch.get(j), corpus, numSteps);
            xNDArray.set(new NDIndex(j), manager.create(X.stream().mapToInt(Integer::intValue).toArray()));
            ArrayList<Integer> Y = data(initialIndicesPerBatch.get(j)+1, corpus, numSteps);
            yNDArray.set(new NDIndex(j), manager.create(Y.stream().mapToInt(Integer::intValue).toArray()));
        }
        NDList pair = new NDList();
        pair.add(xNDArray);
        pair.add(yNDArray);
        pairs.add(pair);
    }
    return pairs;
}

ArrayList<Integer> data(int pos, List<Integer> corpus, int numSteps) {
    // Return a sequence of length `numSteps` starting from `pos`
    return new ArrayList<Integer>(corpus.subList(pos, pos + numSteps));
}

/** 
 * Generate a minibatch of subsequences using sequential partitioning.
 */
public ArrayList<NDList> seqDataIterSequential(List<Integer> corpus, int batchSize, int numSteps, NDManager manager) {
    // Start with a random offset to partition a sequence
    int offset = new Random().nextInt(numSteps);
    int numTokens = ((corpus.size() - offset - 1) / batchSize) * batchSize;
    
    NDArray Xs = manager.create(
        corpus.subList(offset, offset + numTokens).stream().mapToInt(Integer::intValue).toArray());
    NDArray Ys = manager.create(
        corpus.subList(offset + 1, offset + 1 + numTokens).stream().mapToInt(Integer::intValue).toArray());
    Xs = Xs.reshape(new Shape(batchSize, -1));
    Ys = Ys.reshape(new Shape(batchSize, -1));
    int numBatches = (int) Xs.getShape().get(1) / numSteps;
    
    
    ArrayList<NDList> pairs = new ArrayList<NDList>();
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