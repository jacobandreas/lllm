package lllm.main

import java.io.{FileWriter, BufferedWriter}

import breeze.linalg.{SparseVector, DenseVector, Counter, axpy}
import breeze.stats.distributions.Rand
import breeze.util.Index
import erector.corpus.TextCorpusReader
import igor.experiment.{ResultCache, Stage}
import lllm.features.{FrequentSuffixPreprocessor, ConstantFeaturizer, NGramFeaturizer, WordAndIndexFeaturizer}

/**
 * @author jda
 */
object PrepFeaturesSN extends Stage[SelfNormalizingParams] {

  override def run(implicit config: SelfNormalizingParams, cache: ResultCache): Unit = {
    println(config.numLines)
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

    val numNGrams = corpus.nGramIterator(config.order).length

    val sampledFeats = Rand.randInt(contextFeatIndex.size).sample(100)
    val contextFeatureMeans = Counter(contextFeatureCounts.toMap.map { case (feat, count) => contextFeatIndex.indexOf(feat) -> count / numNGrams })
    val contextFeatureMeanVec = DenseVector[Double]((0 until contextFeatIndex.size).map { i => contextFeatureMeans(i) }:_*)
    val covariances = IndexedSeq.fill(100)(DenseVector.zeros[Double](contextFeatIndex.size))
    corpus.nGramIterator(config.order).foreach { nGram =>
      val feats = contextFeaturizer(nGram).map(contextFeatIndex).filterNot(_ == -1).toSet
      val featsHere = SparseVector[Double](contextFeatIndex.size)(feats.map(_ -> 1d).toSeq:_*)
      val covHere = featsHere - contextFeatureMeanVec
      sampledFeats.zipWithIndex.filter(feats contains _._1).foreach { case (featId, rId) =>
        axpy(1 - contextFeatureMeans(featId), covHere, covariances(rId))
      }
    }

//    val normalized: IndexedSeq[DenseVector[Double]] = covariances.map(c => (c * (1d/numNGrams)))
//    val sorted = normalized.map(n => n.toArray.sorted.reverse.toIndexedSeq.grouped(100).map(_.head))
//    val writer = new BufferedWriter(new FileWriter("cov_profiles.csv"))
//    sorted.foreach{s => writer.write(s.mkString(",")); writer.newLine()}
//    System.exit(1)

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
