package lllm.old.model

import breeze.features.FeatureVector
import breeze.linalg._
import breeze.numerics._
import breeze.util.Index
import igor.experiment.ResultCache
import lllm.features.Featurizer
import lllm.main.CrossProductIndex
import lllm.old.main.LLLMParams

/**
 * @author jda
 */
case object Exact extends Objective {

  override def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int = numFeatures

  override def apply(theta: DenseVector[Double],
                     batch: Int,
                     batchLines: Seq[String],
                     featureIndex: Index[String])
                    (implicit config: LLLMParams,
                     cache: ResultCache): (Double, DenseVector[Double]) = {
    var ll = 0d
    val grad = DenseVector.zeros[Double](theta.length)

    // TODO(jda) everyone knows it's a CrossProductIndex
    val cpFeatureIndex = featureIndex.asInstanceOf[CrossProductIndex]
    val vocabIndex: Index[String] = cache.get('VocabIndex)
    val batchContextSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"ContextFeatures$batch"))
    val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
    val predictionFeaturizer: Featurizer = cache.getDisk('PredictionFeaturizer)

    val vocabPredFeats: Array[Array[Int]] = vocabIndex.map(w => predictionFeaturizer(IndexedSeq(w)).map(vocabIndex).toArray).toArray

    (batchContextSamples zip batchDataSamples) foreach { case (context, data) =>
      val fvData = new FeatureVector(data)
      val score = theta dot fvData
      ll += score
      axpy(1d, fvData, grad)

      val denomScores = new Array[Double](vocabPredFeats.size)
      var i = 0
      while (i < denomScores.length) {
        val denomFeats = cpFeatureIndex.crossProduct(vocabPredFeats(i), context)
        denomScores(i) = exp(theta dot new FeatureVector(denomFeats))
        i += 1
      }
      val norm = sum(denomScores)
      ll -= log(norm)
      i = 0
      while (i < denomScores.length) {
        // TODO(jda) better to pre-cache this?
        val denomFeats = cpFeatureIndex.crossProduct(vocabPredFeats(i), context)
        axpy(-denomScores(i) / norm, new FeatureVector(denomFeats), grad)
        i += 1
      }
    }
    (ll, grad)
  }

}
