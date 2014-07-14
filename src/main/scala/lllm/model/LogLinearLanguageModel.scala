package lllm.model

import breeze.linalg.{softmax, sum, DenseVector}
import lllm.features.{Featurizer, WordAndIndexFeaturizer}
import breeze.util.Index
import breeze.features.FeatureVector
import breeze.numerics.{log, exp}
import igor.logging.Logging

/**
 * @author jda
 */
class LogLinearLanguageModel(featurizer: Featurizer,
                             featureIndex: Index[String],
                             vocabIndex: Index[String],
                             val theta: DenseVector[Double]) extends LanguageModel with Serializable {

  def logProb(ngram: IndexedSeq[String]): Double = {
    val logNumerator = new FeatureVector(featurizer(ngram).map(featureIndex).toArray) dot theta
    //val logDenominator = sum(vocabIndex.map { w => exp(new FeatureVector(featurizer(ngram.take(ngram.length-1) :+ w).map(featureIndex).toArray) dot theta)})
    val logDenominator = softmax(vocabIndex.map { w => new FeatureVector(featurizer(ngram.take(ngram.length-1) :+ w).map(featureIndex).toArray) dot theta })
    //println(s"$logNumerator, $logDenominator")
    logNumerator - logDenominator
  }

  def prob(ngram: IndexedSeq[String]): Double = exp(logProb(ngram))

}
