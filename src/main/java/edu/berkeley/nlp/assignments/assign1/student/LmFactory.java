package edu.berkeley.nlp.assignments.assign1.student;

import breeze.linalg.DenseVector;
import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.LanguageModelFactory;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.math.SloppyMath;
import edu.berkeley.nlp.util.StringIndexer;
import lllm.main.SelfNormalizingExperiment;
import sun.reflect.generics.reflectiveObjects.NotImplementedException;

import java.io.*;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * @author jda
 */
public class LmFactory implements LanguageModelFactory {

    public static final int NUM_LINES = 300000;

    public static final double PARAM_L2 = 0.1;
    public static final double PARTITION_L2 = .1;
    public static final double PARTITION_FRAC = 0.01;

    @Override
    public NgramLanguageModel newLanguageModel(final Iterable<List<String>> trainingData) {
        SelfNormalizingExperiment.main(new String[] {
                "--trainPath", "data/training.en.gz",
                "--experimentPath", "work",
                "--numLines", "" + NUM_LINES,
                "--paramL2", "" + PARAM_L2,
                "--partitionL2", "" + PARTITION_L2,
                "--partitionFrac", "" + PARTITION_FRAC
        });

        final Iterable<List<String>> shortTraining = new Iterable<List<String>>() {
            @Override
            public Iterator<List<String>> iterator() {
                final Iterator<List<String>> trainingIter = trainingData.iterator();
                return new Iterator<List<String>>() {
                    int counter = 0;
                    @Override
                    public boolean hasNext() {
                        return counter < NUM_LINES && trainingIter.hasNext();
                    }
                    @Override
                    public List<String> next() {
                        counter++;
                        return trainingIter.next();
                    }
                    @Override
                    public void remove() {
                        throw new NotImplementedException();
                    }
                };
            }
        };
        NgramLanguageModel realLM = null;
        try {
            realLM = (NgramLanguageModel)(new ObjectInputStream(new FileInputStream("optModel.ser")).readObject());
            double trainPartition = computePartition("data/training.en.gz", realLM);
            double testPartition = computePartition("data/test.en", realLM);
            System.out.println("AVERAGE TRAIN PARTITION " + trainPartition);
            System.out.println("AVERAGE TEST PARTITION " + testPartition);
        } catch (Exception e) {
            e.printStackTrace();
        }
        final NgramLanguageModel realFinalLM = realLM;
        final NgramLanguageModel baseLM = new BaseLmFactory().newLanguageModel(shortTraining);

        return new NgramLanguageModel() {
            @Override
            public int getOrder() {
                return 3;
            }

            @Override
            public double getNgramLogProbability(int[] ngram, int from, int to) {
                double realScore = realFinalLM.getNgramLogProbability(ngram, from, to);
                if (realScore != Double.NEGATIVE_INFINITY) {
                    return realScore;
                } else {
                    return baseLM.getNgramLogProbability(ngram, from, to);
                }
            }

            @Override
            public long getCount(int[] ngram) {
                return 0;
            }
        };
    }

    private static double computePartition(String path, NgramLanguageModel lm) throws IOException {
        double total = 0;
        int contextCount = 0;
        int lineNo = 0;
        BufferedReader reader;
        if (path.endsWith(".gz")) {
            reader = new BufferedReader(new InputStreamReader(new GZIPInputStream(new FileInputStream(path))));
        } else {
            reader = new BufferedReader(new FileReader(path));
        }
        while (lineNo++ < 300) {
            String line = reader.readLine();
            String[] words = line.trim().split(" ");
            for (int i = 0; i < words.length-1; i++) {
                contextCount++;
                double part = spotCheckContextNormalizes(lm, Arrays.copyOfRange(words, i, i+2));
                if (contextCount % 100 == 0) System.out.println(part);
                total += part;
            }
        }
        return total / contextCount;
    }

    private static double spotCheckContextNormalizes(NgramLanguageModel languageModel, String[] rawContext) {
        StringIndexer indexer = EnglishWordIndexer.getIndexer();
        int[] context = index(rawContext);
        int[] ngram = new int[context.length + 1];
        for (int i = 0; i < context.length; i++) {
            ngram[i] = context[i];
        }
        double totalLogProb = Double.NEGATIVE_INFINITY;
        for (int wordIdx = 0; wordIdx < indexer.size(); wordIdx++) {
            ngram[ngram.length - 1] = wordIdx;
            totalLogProb = SloppyMath.logAdd(totalLogProb, languageModel.getNgramLogProbability(ngram, 0, ngram.length));
        }
        return Math.abs(totalLogProb);
    }

    private static int[] index(String[] arr) {
        int[] indexedArr = new int[arr.length];
        for (int i = 0; i < indexedArr.length; i++) {
            // indexedArr[i] = EnglishWordIndexer.getIndexer().addAndGetIndex(arr[i]);
            indexedArr[i] = EnglishWordIndexer.getIndexer().indexOf(arr[i]);
        }
        return indexedArr;
    }
}
