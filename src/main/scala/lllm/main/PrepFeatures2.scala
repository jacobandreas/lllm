package lllm.main

import breeze.linalg.Counter
import breeze.util.Index
import erector.corpus.{TextCorpusReader, TextCorpusReader2}
import igor.experiment.{ResultCache, Stage}
import lllm.features._
import lllm.util.HuffmanDict

import scala.collection.mutable

/**
 * @author jda
 */
object PrepFeatures2 extends Stage[LLLMParams2] {

  override def run(implicit config: LLLMParams2, cache: ResultCache): Unit = {

    val corpus = TextCorpusReader(config.trainPath).prefix(config.nLines)
    //val corpus = TextCorpusReader2(config.trainPath)

    val vocabIndex = corpus.vocabularyIndex
    //val unigramCounts = corpus.nGramIterator(1).map(unigram => unigram(0) -> 1d).toMap
    val unigramCounts = Counter(corpus.nGramIterator(1).map(unigram => unigram(0) -> 1d)).toMap.filter { case (word, count) => count > config.minWordCount }

    logger.info(s"vocabulary size is ${unigramCounts.size} / ${vocabIndex.size}")

    //val contextFeaturizer = (FrequentSuffixPreprocessor(Counter(unigramCounts), config.minSuffixCount, "UNK") before NGramFeaturizer(2, config.order)) + ConstantFeaturizer()
    //val contextFeaturizer = NGramFeaturizer(2, config.order) + ConstantFeaturizer()
    //val contextFeaturizer = WordAndIndexFeaturizer() + NGramFeaturizer(2, config.order) + ConstantFeaturizer()
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
    //val positiveContextFeatIndex = Index(corpus.nGramFeatureIndexer(config.order, contextFeaturizer(_)))
    val contextFeatIndex = positiveContextFeatIndex

    val ctxMap = contextFeatureCounts.toMap
    logger.info("count 1 features: " + ctxMap.count(_._2 == 1))
    logger.info("count 2 features: " + ctxMap.count(_._2 == 2))
    logger.info("count >=3 features: " + ctxMap.count(_._2 >= 3))

    cache.put('ContextFeaturizer, contextFeaturizer)
    cache.put('ContextFeatureIndex, contextFeatIndex)
    cache.put('VocabularyIndex, vocabIndex)

    val huffmanDict = task("building Huffman dictionary") {
      val wordIdsAndCounts = unigramCounts.map { case (word, count) => corpus.vocabularyIndex(word) -> count }
      HuffmanDict.fromCounts(wordIdsAndCounts)
    }

    cache.put('HuffmanDict, huffmanDict)

    val cpIndexBuilder = new CrossProductIndex.Builder(huffmanDict.prefixIndex,
                                                       contextFeatIndex,
                                                       hashFeatures = 0)

    corpus.nGramIterator(config.order).foreach { ngram =>
      val prediction = ngram.last
      val predictionId = vocabIndex(prediction)
      val optCode = huffmanDict.dict.get(predictionId)
      if (optCode.isDefined) {
        val code = optCode.get
        val context = contextFeaturizer(ngram)
        val contextIds = context map contextFeatIndex
        code.tails.toArray.foreach { codePrefix =>
          val decisionIds = Array(huffmanDict.prefixIndex(codePrefix), huffmanDict.constIndex)
          cpIndexBuilder.add(decisionIds, contextIds)
        }
      }
    }

//    val featureCounter = Counter[Int,Double](corpus.nGramIterator(config.order).flatMap { ngram =>
//      val prediction = ngram.last
//      val predictionId = vocabIndex(prediction)
//      val optCode = huffmanDict.dict.get(predictionId)
//      if (optCode.isDefined) {
//        val code = optCode.get
//        val context = contextFeaturizer(ngram)
//        val contextIds = context map contextFeatIndex
//        code.tails.toArray.flatMap { codePrefix =>
//          val decisionIds = Array(huffmanDict.prefixIndex(codePrefix), huffmanDict.constIndex)
//          cpIndexBuilder.add(decisionIds, contextIds).map(_ -> 1d)
//        }
//      } else {
//        None
//      }
//    })

//    val initProductIndex = cpIndexBuilder.result()
//    val productIndex: CrossProductIndex = initProductIndex.filter(k => featureCounter(initProductIndex(k)) >= config.minFeatureCount)
//    logger.info(s"product index size is ${productIndex.size}")

    val productIndex = cpIndexBuilder.result()
    cache.put('ProductIndex, productIndex)
    cache.put('NLineGroups, Int.box(corpus.lineGroupIterator(config.lineGroupSize).length))

    cache.writeFile('Vocab, unigramCounts.keySet)
  }

}
