public class TimeMachine {
    // Split text lines into word or character tokens.
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

    // Read `The Time Machine` dataset and return an array of the lines
    public static String[] readTimeMachine() throws IOException {
        URL url = new URL("http://d2l-data.s3-accelerate.amazonaws.com/timemachine.txt");
        BufferedReader in = new BufferedReader(new InputStreamReader(url.openStream()));
        Object[] linesObjects = in.lines().toArray();

        String[] lines = new String[linesObjects.length];
        for (int i = 0; i < linesObjects.length; i++) {
            lines[i] = ((String) linesObjects[i]).replaceAll("[^A-Za-z]+", " ").strip().toLowerCase();
        }
        return lines;
    }

    public static Object[] loadCorpusTimeMachine(int maxTokens) throws IOException, Exception {
        /* Return token indices and the vocabulary of the time machine dataset. */
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
        return new Object[] {corpus, vocab};
    }

}

public class Vocab {
    public int unk;
    public List<Map.Entry<String, Integer>> tokenFreqs;
    public List<String> idxToToken;
    public HashMap<String, Integer> tokenToIdx;

    public Vocab(String[][] tokens, int minFreq, String[] reservedTokens) {
        // Sort according to frequencies
        HashMap<String, Integer> counter = countCorpus2D(tokens);
        this.tokenFreqs = new LinkedList<Map.Entry<String, Integer>>(counter.entrySet()); 
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

public HashMap<String, Integer> countCorpus(String[] tokens) {
    /* Count token frequencies. */
    HashMap<String, Integer> counter = new HashMap<>();
    if (tokens.length != 0) {
        for (String token : tokens) {
            counter.put(token, counter.getOrDefault(token, 0)+1);
        }
    }
    return counter;
}

public HashMap<String, Integer> countCorpus2D(String[][] tokens) {
    /* Flatten a list of token lists into a list of tokens */
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