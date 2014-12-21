package edu.berkeley.nlp.assignments.assign1.student;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.langmodel.NgramLanguageModel;
import edu.berkeley.nlp.util.StringIndexer;
import struct.CopycatHashMap;
import struct.OpenHashMap;

/**
 * @author jda
 */
public class TrigramLanguageModel implements NgramLanguageModel {

    public static final double DISCOUNT = 0.1;

    private final int[] unigramTypeCounts;
    private final int unigramTypeNormalizer;
    private final CopycatHashMap bigramTypeCounts;
    private final int[] bigramTypeNormalizers;
    private final int[] bigramCountsInContext;
    private final OpenHashMap trigramTokenCounts;
    private final OpenHashMap trigramTokenNormalizers;
    private final CopycatHashMap trigramCountsInContext;
    private final StringIndexer vocabIndex;

    private final int START_INDEX;

    public TrigramLanguageModel(int[] unigramTypeCounts,
                                int unigramTypeNormalizer,
                                CopycatHashMap bigramTypeCounts,
                                int[] bigramTypeNormalizers,
                                int[] bigramCountsInContext,
                                OpenHashMap trigramTokenCounts,
                                OpenHashMap trigramTokenNormalizers,
                                CopycatHashMap trigramCountsInContext) {
        this.unigramTypeCounts = unigramTypeCounts;
        this.unigramTypeNormalizer = unigramTypeNormalizer;
        this.bigramTypeCounts = bigramTypeCounts;
        this.bigramTypeNormalizers = bigramTypeNormalizers;
        this.bigramCountsInContext = bigramCountsInContext;
        this.trigramTokenCounts = trigramTokenCounts;
        this.trigramTokenNormalizers = trigramTokenNormalizers;
        this.trigramCountsInContext = trigramCountsInContext;
        vocabIndex = EnglishWordIndexer.getIndexer();
        START_INDEX = vocabIndex.indexOf(START);
    }

    @Override
    public int getOrder() {
        return 3;
    }

    @Override
    public double getNgramLogProbability(int[] ngram, int from, int to) {

        double currentProb = 0;

        int predWord = ngram[to-1];

        for (int i = 1; i <= to - from; i++) {
            if (i == 1) {
                // unigram model
                if (predWord >= unigramTypeCounts.length || predWord == START_INDEX) {
                    // OOV
                    return 0;
                }
                currentProb = 1d * unigramTypeCounts[predWord] / unigramTypeNormalizer;
            } else if (i == 2) {
                long bigramKey = BaseLmFactory.pack(ngram, to-2, to);
                int contextWord = ngram[to-2];
                if (contextWord > unigramTypeCounts.length) {
                    break;
                }
                double denom = bigramTypeNormalizers[contextWord];
                if (denom == 0) {
                    break;
                }
                double count = bigramTypeCounts.get(bigramKey);
                double disCount = Math.max(count - DISCOUNT, 0);
                double discountProb = disCount / denom;
                double smoother = DISCOUNT * bigramCountsInContext[contextWord] / denom;
                currentProb = discountProb + smoother * currentProb;
            } else {
                long trigramKey = BaseLmFactory.pack(ngram, to-3, to);
                long contextKey = BaseLmFactory.pack(ngram, to-3, to-1);
                double denom = trigramTokenNormalizers.get(contextKey);
                if (denom == -1) {
                    break;
                }
                double count = trigramTokenCounts.get(trigramKey);
                double disCount = Math.max(count - DISCOUNT, 0);
                double discountProb = disCount / denom;
                double smoother = DISCOUNT * trigramCountsInContext.get(contextKey) / denom;
                currentProb = discountProb + smoother * currentProb;
            }
        }


        double logProb = Math.log(currentProb);

//        System.out.println(vocabIndex.get(predWord));
//        System.out.println(unigramTypeCounts[predWord]);
//        System.out.println();

        assert(!Double.isInfinite(logProb));
        assert(!Double.isNaN(logProb));
        assert(logProb < 0);
        return logProb;

    }

    @Override
    public long getCount(int[] ngram) {
        return 0;
    }
}
