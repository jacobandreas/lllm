package lllm.main

import breeze.linalg.Counter
import breeze.util.Index
import erector.corpus.TextCorpusReader
import igor.experiment.{ResultCache, Stage}
import lllm.features.{FrequentSuffixPreprocessor, ConstantFeaturizer, NGramFeaturizer, WordAndIndexFeaturizer}

/**
 * @author jda
 */
object PrepFeaturesSN extends Stage[SelfNormalizingParams] {

  override def run(implicit config: SelfNormalizingParams, cache: ResultCache): Unit = {
    val corpus = TextCorpusReader(config.trainPath).prefix(config.numLines)
//    val vocabIndex = corpus.vocabularyIndex
    val unigramCounts = Counter(corpus.nGramIterator(1).map(unigram => unigram(0) -> 1d)).toMap.filter { case (word, count) => count > config.minWordCount }
    val vocabIndex = Index(unigramCounts.keys)
    logger.info(s"vocabulary size is ${unigramCounts.size} / ${vocabIndex.size}")

    val baseContextFeaturizer = if (config.useNGramFeatures && config.useSkipGramFeatures) {
      WordAndIndexFeaturizer() + NGramFeaturizer(config.nGramMinOrder, config.nGramMaxOrder) + ConstantFeaturizer()
    } else if (config.useNGramFeatures) {
      NGramFeaturizer(config.nGramMinOrder, config.nGramMaxOrder) + ConstantFeaturizer()
    } else if (config.useSkipGramFeatures) {
      WordAndIndexFeaturizer() + ConstantFeaturizer()
    } else {
      ConstantFeaturizer()
    }
    val contextFeaturizer = FrequentSuffixPreprocessor(Counter(unigramCounts), config.minSuffixCount, "UNK") before baseContextFeaturizer
    val contextFeatureCounts = Counter(corpus.nGramIterator(config.order).flatMap { ngram => contextFeaturizer(ngram).map(_ -> 1d) })
    val positiveContextFeatIndex = Index(corpus.nGramFeatureIndexer(config.order, contextFeaturizer(_)).filter(contextFeatureCounts(_) >= config.minFeatureCount))
    val contextFeatIndex = positiveContextFeatIndex

    val ctxMap = contextFeatureCounts.toMap
    logger.info("count 1 features: " + ctxMap.count(_._2 == 1))
    logger.info("count 2 features: " + ctxMap.count(_._2 == 2))
    logger.info("count >=3 features: " + ctxMap.count(_._2 >= 3))

    cache.put('contextFeaturizer, contextFeaturizer)
    cache.put('contextFeatureIndex, contextFeatIndex)
    cache.put('vocabularyIndex, vocabIndex)

    val cpIndexBuilder = new CrossProductIndex.Builder(vocabIndex, contextFeatIndex)
    corpus.nGramIterator(config.order).foreach { nGram =>
      if (vocabIndex contains nGram.last) {
        val prediction = nGram.last
        val predictionId = vocabIndex(prediction)
        val context = contextFeaturizer(nGram)
        val contextIds = context map contextFeatIndex
        cpIndexBuilder.add(Array(predictionId), contextIds)
      }
    }

    val productIndex = cpIndexBuilder.result()
    cache.put('productIndex, productIndex)
    cache.put('numLineGroups, Int.box(corpus.lineGroupIterator(config.lineGroupSize).length))
    cache.writeFile('vocab, unigramCounts.keySet)
  }

}
