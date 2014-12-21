package edu.berkeley.nlp.assignments.assign1.student;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.util.StringIndexer;
import struct.OpenHashMap;

import java.util.Arrays;

/**
 * @author jda
 */
public class TrigramKNLanguageModel implements NgramLanguageModel {

    public static final int ORDER = 3;
    public static final int BITS_PER_KEY_PART = 20;

    public static final double DISCOUNT = 0.1;

    final int[] unigramTokenCounts;
    final int[] unigramTypeCounts;
    final OpenHashMap[] ngramTokenCounts;
    final OpenHashMap[] ngramTypeCounts;
    final int[] ngramTypeNormalizers;
    final int unigramTypeNormalizer;

    final StringIndexer vocabIndex;

    public TrigramKNLanguageModel(int[] unigramTokenCounts,
                                  int[] unigramTypeCounts,
                                  OpenHashMap[] ngramTokenCounts,
                                  OpenHashMap[] ngramTypeCounts,
                                  int[] ngramTypeNormalizers) {
        this.unigramTokenCounts = unigramTokenCounts;
        this.unigramTypeCounts = unigramTypeCounts;
        this.ngramTokenCounts = ngramTokenCounts;
        this.ngramTypeCounts = ngramTypeCounts;
        this.vocabIndex = EnglishWordIndexer.getIndexer();

        this.ngramTypeNormalizers = ngramTypeNormalizers;
        int unigramTypeNormalizer = 0;
        for (int i = 0; i < unigramTokenCounts.length; i++) {
            unigramTypeNormalizer += unigramTypeCounts[i];
        }
        this.unigramTypeNormalizer = unigramTypeNormalizer;
    }

    @Override
    public int getOrder() {
        return ORDER;
    }

    @Override
    public double getNgramLogProbability(int[] ngram, int from, int to) {
        int[] cpGram = Arrays.copyOfRange(ngram, from, to);
        int[] cpContext = Arrays.copyOfRange(ngram, from, to-1);

        int order = to - from;
        double numCount;
        double denomCount;

        if (order == 3) {
            numCount = ngramTokenCounts[order].get(pack(cpGram)) - DISCOUNT;
            denomCount = ngramTokenCounts[order-1].get(pack(cpContext));
        } else if (order == 2) {
            numCount = ngramTypeCounts[order].get(pack(cpGram)) - DISCOUNT;
            // TODO(jda) this is wrong!
//            System.out.println(cpContext[0]);
//            System.out.println(unigramTypeCounts.length);
//            System.out.println(vocabIndex.size());
//            System.out.println(vocabIndex.get(vocabIndex.size()-1));
            if (cpContext[0] <= unigramTypeCounts.length) {
                denomCount = ngramTypeNormalizers[cpContext[0]];
            } else {
                denomCount = -1;
            }
        } else {
            assert (order == 1);
            if (cpGram[0] >= unigramTypeCounts.length) {
                numCount = -1;
                denomCount = -1;
            } else {
                numCount = unigramTypeCounts[cpGram[0]];
                denomCount = unigramTypeNormalizer;
            }
        }

        double result;
        if (order == 1 && numCount == -1) {
            // out of vocab
            System.out.println(Arrays.toString(cpGram) + " OOV");
            result = 0;
        } else if (numCount == -1 && denomCount == -1) {
            // back off
            System.out.println(Arrays.toString(cpGram) + " back off");
            result = getNgramLogProbability(cpGram, from + 1, to);
        } else if (order == 1) {
            // use unigram dist
            System.out.println(Arrays.toString(cpGram) + " unigram");
            System.out.println(vocabIndex.get(cpGram[0]));

            assert(numCount > 0);
            result = Math.log(1d * numCount / denomCount);
        } else {
            // use smoothing
            System.out.println(Arrays.toString(cpGram) + " smooth");
//            System.out.println(numCount);
//            System.out.println(denomCount);
//            System.out.println((numCount - DISCOUNT) / denomCount);
//            System.out.println();
            System.out.println(Math.log(Math.max(numCount - DISCOUNT, 0)) / denomCount);

            result = Math.log(Math.max(numCount - DISCOUNT, 0) / denomCount +
                              DISCOUNT * unigramTokenCounts.length / denomCount * Math.exp(getNgramLogProbability(ngram, from+1, to)));
        }

        System.out.println(result);
        assert(result <= 0 && !Double.isInfinite(result) && !Double.isNaN(result));
        return result;

        //return -Double.MAX_VALUE;
        //return 0;

//        int numCount = ngramTokenCounts[cpGram.length].get(pack(cpGram));
//        int denomCount;
//
//        if (cpContext.length >= 2) {
//            denomCount = ngramTokenCounts[cpGram.length-1].get(pack(cpContext));
//        } else {
//            denomCount = unigramTokenCounts[cpContext[0]];
//        }
//
//        if (numCount == -1 && denomCount == -1) {
//            return -Double.MAX_VALUE;
//        }
//
//        if (numCount == -1) {
//            return -Double.MAX_VALUE;
//        }

//                System.out.println();
//                System.out.println(ngramTokenCounts[cpGram.length].size());
//                System.out.println(ngramTokenCounts[cpGram.length].capacity());
//                System.out.println(Arrays.toString(ngram));
//                System.out.println(numCount);
//                System.out.println(denomCount);

        //return Math.log(numCount) - Math.log(denomCount);
    }

    @Override
    public long getCount(int[] ngram) {
        if (ngram.length == 1) {
            if (!vocabIndex.contains(ngram)) return 0;
            return unigramTokenCounts[ngram[0]];
        }
        long key = pack(ngram);
        int count = ngramTokenCounts[ngram.length].get(key);
        if (count == -1) return 0;
        return count;
    }

    public static long pack(int[] ngram) {
        assert(ngram.length <= 3);
        long r = 1;
        for (int i = 0; i < ngram.length; i++) {
            long idx = ngram[i];
            long shifted = idx << (1 + i * BITS_PER_KEY_PART);
//            System.out.println();
//            System.out.println(idx);
//            System.out.println(r);
//            System.out.println(shifted);
            assert((r & shifted) == 0);
            r += shifted;
        }
        return r;
    }

    public static long pack(String[] ngram, StringIndexer vocabIndex) {
        int[] keys = new int[ngram.length];
        for (int i = 0; i < keys.length; i++) {
            keys[i] = vocabIndex.indexOf(ngram[i]);
        }
        return pack(keys);
    }



}
