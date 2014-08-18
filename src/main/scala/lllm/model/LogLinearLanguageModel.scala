package lllm.model

import breeze.features.FeatureVector
import breeze.linalg.{DenseVector, softmax}
import breeze.numerics.{log1p, exp, Inf}
import lllm.features.Featurizer
import lllm.main.CrossProductIndex
import lllm.util.HuffmanDict

/**
 * @author jda
 */
class LogLinearLanguageModel(predictionFeaturizer: Featurizer,
                             contextFeaturizer: Featurizer,
                             index: CrossProductIndex,
                             val theta: DenseVector[Double]) extends LanguageModel with Serializable {

  def logProb(ngram: IndexedSeq[String]): Double = {
    val contextFeatures = contextFeaturizer(ngram).flatMap(index.secondIndex.indexOpt).toArray
    // TODO(jda) use prediction featurizer
    if (!index.firstIndex.contains(ngram.last)) return -Inf
    val predWord: Int = index.firstIndex(ngram.last)
    // TODO(jda) currently makes the assumption that vocabIndex == predictionFeatureIndex
    val logNumerator = score(contextFeatures, predWord)
    val logDenominator = softmax(DenseVector.tabulate(index.firstIndex.size){score(contextFeatures, _)})
    logNumerator - logDenominator
  }

  def score(contextFeatures: Array[Int], word: Int) = {
    theta dot new FeatureVector(index.crossProduct(Array(word), contextFeatures))
  }

  def prob(ngram: IndexedSeq[String]): Double = exp(logProb(ngram))

}

class HierarchicalLanguageModel(predictionFeaturizer: Featurizer,
                                contextFeaturizer: Featurizer,
                                index: CrossProductIndex,
                                huffmanDict: HuffmanDict[Int],
                                val theta: DenseVector[Double]) extends LanguageModel with Serializable {
  def logProb(ngram: IndexedSeq[String]): Double = {
    val contextFeatures = contextFeaturizer(ngram).flatMap(index.secondIndex.indexOpt).toArray
    if (!index.firstIndex.contains(ngram.last)) return -Inf
    val wordId: Int = index.firstIndex(ngram.last)
    val code = huffmanDict.dict.get(wordId).get
    var score = 0d
    // TODO(jda) de-HOF-ify
    code.tails.toArray.filter(_.nonEmpty).foreach { prefix =>
      val decision = if (prefix.head) 1d else -1d
      val history = prefix.tail
      val nodeId = huffmanDict.prefixIndex(history)
      val feats = index.crossProduct(Array(nodeId), contextFeatures)
      score += -log1p(exp((theta dot new FeatureVector(feats)) * decision))
    }
    //println("giving back", score)
    score
  }

  def prob(ngram: IndexedSeq[String]): Double = exp(logProb(ngram))
}
