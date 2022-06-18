import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.util.DownloadUtils;
import ai.djl.util.Pair;

import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Enumeration;
import java.util.HashSet;
import java.util.stream.IntStream;
import java.util.zip.ZipEntry;
import java.util.zip.ZipFile;

public class NMT {

    public static String readDataNMT() throws IOException {
        DownloadUtils.download(
                "http://d2l-data.s3-accelerate.amazonaws.com/fra-eng.zip", "fra-eng.zip");
        ZipFile zipFile = new ZipFile(new File("fra-eng.zip"));
        Enumeration<? extends ZipEntry> entries = zipFile.entries();
        while (entries.hasMoreElements()) {
            ZipEntry entry = entries.nextElement();
            if (entry.getName().contains("fra.txt")) {
                InputStream stream = zipFile.getInputStream(entry);
                return new String(stream.readAllBytes(), StandardCharsets.UTF_8);
            }
        }
        return null;
    }

    public static String preprocessNMT(String text) {
        // Replace non-breaking space with space, and convert uppercase letters to
        // lowercase ones

        text = text.replace('\u202f', ' ').replaceAll("\\xa0", " ").toLowerCase();

        // Insert space between words and punctuation marks
        StringBuilder out = new StringBuilder();
        Character currChar;
        for (int i = 0; i < text.length(); i++) {
            currChar = text.charAt(i);
            if (i > 0 && noSpace(currChar, text.charAt(i - 1))) {
                out.append(' ');
            }
            out.append(currChar);
        }
        return out.toString();
    }

    public static boolean noSpace(Character currChar, Character prevChar) {
        /* Preprocess the English-French dataset. */
        return new HashSet<>(Arrays.asList(',', '.', '!', '?')).contains(currChar)
                && prevChar != ' ';
    }

    public static Pair<ArrayList<String[]>, ArrayList<String[]>> tokenizeNMT(
            String text, Integer numExamples) {
        ArrayList<String[]> source = new ArrayList<>();
        ArrayList<String[]> target = new ArrayList<>();

        int i = 0;
        for (String line : text.split("\n")) {
            if (numExamples != null && i > numExamples) {
                break;
            }
            String[] parts = line.split("\t");
            if (parts.length == 2) {
                source.add(parts[0].split(" "));
                target.add(parts[1].split(" "));
            }
            i += 1;
        }
        return new Pair<>(source, target);
    }

    public static int[] truncatePad(Integer[] integerLine, int numSteps, int paddingToken) {
        /* Truncate or pad sequences */
        int[] line = Arrays.stream(integerLine).mapToInt(i -> i).toArray();
        if (line.length > numSteps) {
            return Arrays.copyOfRange(line, 0, numSteps);
        }
        int[] paddingTokenArr = new int[numSteps - line.length]; // Pad
        Arrays.fill(paddingTokenArr, paddingToken);

        return IntStream.concat(Arrays.stream(line), Arrays.stream(paddingTokenArr)).toArray();
    }

    public static Pair<NDArray, NDArray> buildArrayNMT(
            ArrayList<String[]> lines, Vocab vocab, int numSteps, NDManager manager) {
        /* Transform text sequences of machine translation into minibatches. */
        ArrayList<Integer[]> linesIntArr = new ArrayList<>();
        for (String[] strings : lines) {
            linesIntArr.add(vocab.getIdxs(strings));
        }
        for (int i = 0; i < linesIntArr.size(); i++) {
            ArrayList<Integer> temp = new ArrayList<>(Arrays.asList(linesIntArr.get(i)));
            temp.add(vocab.getIdx("<eos>"));
            linesIntArr.set(i, temp.toArray(new Integer[0]));
        }

        NDArray arr = manager.create(new Shape(linesIntArr.size(), numSteps), DataType.INT32);
        int row = 0;
        for (Integer[] line : linesIntArr) {
            NDArray rowArr = manager.create(truncatePad(line, numSteps, vocab.getIdx("<pad>")));
            arr.set(new NDIndex("{}:", row), rowArr);
            row += 1;
        }
        NDArray validLen = arr.neq(vocab.getIdx("<pad>")).sum(new int[] {1});
        return new Pair<>(arr, validLen);
    }

    public static Pair<ArrayDataset, Pair<Vocab, Vocab>> loadDataNMT(
            int batchSize, int numSteps, int numExamples, NDManager manager) throws IOException {
        /* Return the iterator and the vocabularies of the translation dataset. */
        String text = preprocessNMT(readDataNMT());
        Pair<ArrayList<String[]>, ArrayList<String[]>> pair = tokenizeNMT(text, numExamples);
        ArrayList<String[]> source = pair.getKey();
        ArrayList<String[]> target = pair.getValue();
        Vocab srcVocab =
                new Vocab(
                        source.toArray(new String[0][]),
                        2,
                        new String[] {"<pad>", "<bos>", "<eos>"});
        Vocab tgtVocab =
                new Vocab(
                        target.toArray(new String[0][]),
                        2,
                        new String[] {"<pad>", "<bos>", "<eos>"});

        Pair<NDArray, NDArray> pairArr = buildArrayNMT(source, srcVocab, numSteps, manager);
        NDArray srcArr = pairArr.getKey();
        NDArray srcValidLen = pairArr.getValue();

        pairArr = buildArrayNMT(target, tgtVocab, numSteps, manager);
        NDArray tgtArr = pairArr.getKey();
        NDArray tgtValidLen = pairArr.getValue();

        ArrayDataset dataset =
                new ArrayDataset.Builder()
                        .setData(srcArr, srcValidLen)
                        .optLabels(tgtArr, tgtValidLen)
                        .setSampling(batchSize, true)
                        .build();

        return new Pair<>(dataset, new Pair<>(srcVocab, tgtVocab));
    }
}
