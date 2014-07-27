package lllm.model

import breeze.features.FeatureVector
import breeze.linalg.{DenseVector, softmax}
import breeze.numerics.exp
import lllm.features.Featurizer
import lllm.main.CrossProductIndex

/**
 * @author jda
 */
class LogLinearLanguageModel(predictionFeaturizer: Featurizer,
                             contextFeaturizer: Featurizer,
                             index: CrossProductIndex,
                             val theta: DenseVector[Double]) extends LanguageModel with Serializable {

  def logProb(ngram: IndexedSeq[String]): Double = {
    val contextFeatures = contextFeaturizer(ngram).flatMap(index.secondIndex.indexOpt).toArray
    val predWord: Int = index.firstIndex(ngram.last)
    // TODO: currently makes the assumption that vocabIndex == predictionFeatureIndex
    val logNumerator = score(contextFeatures, predWord)
    val logDenominator = softmax(DenseVector.tabulate(index.firstIndex.size){score(contextFeatures, _)})
    logNumerator - logDenominator
  }

  def score(contextFeatures: Array[Int], word: Int) = {
    theta dot new FeatureVector(index.crossProduct(Array(word), contextFeatures))
  }

  def prob(ngram: IndexedSeq[String]): Double = exp(logProb(ngram))

}
