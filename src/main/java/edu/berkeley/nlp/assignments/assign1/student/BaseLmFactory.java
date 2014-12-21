package edu.berkeley.nlp.assignments.assign1.student;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.LanguageModelFactory;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.util.StringIndexer;
import struct.CopycatHashMap;
import struct.OpenHashMap;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author jda
 */
public class BaseLmFactory implements LanguageModelFactory {

    private static final int ORDER = 3;
    private static final int BITS_PER_KEY_PART = 20;

    @Override
    public NgramLanguageModel newLanguageModel(Iterable<List<String>> trainingData) {

        StringIndexer vocabIndex = makeVocabIndex(trainingData);

        int[] unigramTypeCounts = new int[vocabIndex.size()];
        int unigramTypeNormalizer = 0;

        //12109007, 62479687

        // 15000001, 60000001 is a good pair
        OpenHashMap bigramTokenCounts = new OpenHashMap(17000359);
        //OpenHashMap bigramTypeCounts = new OpenHashMap(15000001);
        CopycatHashMap bigramTypeCounts = new CopycatHashMap(bigramTokenCounts);
        int[] bigramTypeNormalizers = new int[vocabIndex.size()];
        int[] bigramCountsInContext = new int[vocabIndex.size()]; // holds unigrams

        OpenHashMap trigramTokenCounts = new OpenHashMap(61000417);
        //OpenHashMap trigramCountsInContext = new OpenHashMap(15000001); // holds bigrams
        CopycatHashMap trigramCountsInContext = new CopycatHashMap(bigramTokenCounts);

        for (List<String> rawLine : trainingData) {
//            System.out.println(rawLine);
            ArrayList<String> line = new ArrayList<String>(rawLine.size()+2);
            line.add(NgramLanguageModel.START);
            for (String word : rawLine) line.add(word);
            line.add(NgramLanguageModel.STOP);
            int[] indices = getIndices(line, vocabIndex);
            for (int i = 1; i < indices.length; i++) {
                long bigramKey = pack(indices, i-1, i+1);
                boolean newBigram = bigramTokenCounts.increment(bigramKey);
                if (newBigram) {
                    unigramTypeCounts[indices[i]] += 1;
                    bigramCountsInContext[indices[i-1]] += 1;
                }

                long trigramKey = pack(indices, i-2, i+1);
                boolean newTrigram = trigramTokenCounts.increment(trigramKey);
                if (newTrigram) {
                    bigramTypeCounts.increment(bigramKey);
                    bigramTypeNormalizers[indices[i-1]] += 1;
                    long contextKey = pack(indices, i-2, i);
                    trigramCountsInContext.increment(contextKey);
                }
            }
        }
        for (int i = 0; i < vocabIndex.size(); i++) {
            unigramTypeNormalizer += unigramTypeCounts[i];
        }

        return new TrigramLanguageModel(unigramTypeCounts,
                                        unigramTypeNormalizer,
                                        bigramTypeCounts,
                                        bigramTypeNormalizers,
                                        bigramCountsInContext,
                                        trigramTokenCounts,
                                        bigramTokenCounts,
                                        trigramCountsInContext);
    }

    private int[] getIndices(List<String> line, StringIndexer index) {
        int[] r = new int[line.size()];
        for (int i = 0; i < line.size(); i++) {
            r[i] = index.indexOf(line.get(i));
        }
        return r;
    }

    public static long pack(int[] ngram, int from, int to) {
//        System.out.println(Arrays.toString(ngram));
        assert(from - to <= 3);
//        System.out.println();
//        System.out.println("pack " + Arrays.toString(Arrays.copyOfRange(ngram, Math.max(from,0), to)));
        long r = 1;
        for (int i = from; i < to; i++) {
            if (i < 0) continue; // TODO handle differently?
            long idx = ngram[i];
            long shifted = idx << (1 + (i-from) * BITS_PER_KEY_PART);
//            System.out.println(Long.toBinaryString(idx));
//            System.out.println(Long.toBinaryString(shifted));
            assert((r & shifted) == 0);
            r |= shifted;
        }
//        System.out.println(r);
        return r;
    }

//    @Override
//    public NgramLanguageModel newLanguageModel(Iterable<List<String>> trainingData) {
//        final StringIndexer vocabIndex = makeVocabIndex(trainingData);
//        assert (Math.log(vocabIndex.size()) / Math.log(2) < BITS_PER_KEY_PART);
//        final int[] unigramTokenCounts = new int[vocabIndex.size()];
//        final int[] unigramTypeCounts = new int[vocabIndex.size()];
//        final OpenHashMap[] ngramTokenCounts = new OpenHashMap[ORDER + 1];
//        final OpenHashMap[] ngramTypeCounts = new OpenHashMap[ORDER];
//        //final int[] ngramTypeNormalizers = new int[vocabIndex.size()];
//        for (int i = 2; i <= ORDER; i++) {
//            ngramTokenCounts[i] = new OpenHashMap();
//            if (i < ORDER) {
//                ngramTypeCounts[i] = new OpenHashMap();
//            }
//        }
//        collectCounts(trainingData,
//                      unigramTokenCounts,
//                      unigramTypeCounts,
//                      ngramTokenCounts,
//                      ngramTypeCounts,
//                      ngramTypeNormalizers,
//                      vocabIndex);
//        return new TrigramKNLanguageModel(unigramTokenCounts,
//                                          unigramTypeCounts,
//                                          ngramTokenCounts,
//                                          ngramTypeCounts);
//
//    }
//
    private StringIndexer makeVocabIndex(Iterable<List<String>> trainingData) {
        StringIndexer index = EnglishWordIndexer.getIndexer();
        for (List<String> words : trainingData) {
            for (String word : words) {
                index.add(word);
            }
        }
        index.add(NgramLanguageModel.START);
        index.add(NgramLanguageModel.STOP);
        return index;
    }
//
//    private void collectCounts(Iterable<List<String>> trainingData,
//                               int[] unigramTokenCounts,
//                               int[] unigramTypeCounts,
//                               OpenHashMap[] ngramTokenCounts,
//                               OpenHashMap[] ngramTypeCounts,
//                               int[] ngramTypeNormalizers,
//                               StringIndexer vocabIndex) {
//        for (List<String> inWords : trainingData) {
////            System.out.println(words);
//            ArrayList<String> words = new ArrayList<String>(inWords.size() + 2);
//            words.add(NgramLanguageModel.START);
//            for (String w : inWords) {
//                words.add(w);
//            }
//            words.add(NgramLanguageModel.STOP);
//            for (int i = 0; i < words.size() + 1; i++) {
//                if (i < words.size()) {
//                    unigramTokenCounts[vocabIndex.indexOf(words.get(i))] += 1;
//                }
//                int maxOrder = i == words.size() ? 2 : ORDER;
//                for (int o = 2; o <= maxOrder; o++) {
//                    String[] ngram;
//                    if (i == words.size()) {
//                        ngram = new String[] { words.get(words.size() - 2), words.get(words.size() - 1)};
//                    } else {
//                        ngram = getNgram(words, i, o);
//                    }
//                    //System.out.println(Arrays.toString(ngram));
//                    long key = TrigramKNLanguageModel.pack(ngram, vocabIndex);
////                    if (o == 2 && ngram[0].equals("in") && ngram[1].equals("terms")) {
////                        System.out.println(words);
////                        System.out.println(key);
////                        System.out.println(ngramTokenCounts[o].get(key));
////                    }
////                    System.out.println(Arrays.toString(ngram));
//                    if (ngramTokenCounts[o].increment(key)) {
////                        if (o == 2 && ngram[0].equals("in") && ngram[1].equals("terms")) {
////                            System.out.println("NEW");
////                        }
//                        if (o == 2) {
//                            unigramTypeCounts[vocabIndex.indexOf(ngram[1])] += 1;
//                            //System.out.println(ngram[1]);
//                        } else {
//                            String[] prefix = Arrays.copyOfRange(ngram, ORDER-o+1, ngram.length);
//                            ngramTypeCounts[o-1].increment(TrigramKNLanguageModel.pack(prefix, vocabIndex));
//                            assert(o == 3);
//                            ngramTypeNormalizers[vocabIndex.indexOf(ngram[1])] += 1;
//                        }
//                    }
//                }
//            }
//        }
//    }
//
//    // TODO this may or may not be wrong
//    private String[] getNgram(List<String> words, int index, int order) {
//        String[] r = new String[order];
//        for (int i = 0; i < order; i++) {
//            String word;
//            if (index - ORDER + 1 + i < 0) {
//                word = NgramLanguageModel.START;
//            } else {
//                word = words.get(index - ORDER + 1 + i);
//            }
//            r[i] = word;
//        }
//        return r;
//    }

}
