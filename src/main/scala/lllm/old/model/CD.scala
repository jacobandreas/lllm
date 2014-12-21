package lllm.old.model

import breeze.features.FeatureVector
import breeze.linalg._
import breeze.numerics._
import breeze.util.Index
import igor.experiment.ResultCache
import lllm.old.main.LLLMParams

/**
 * @author jda
 */
case object CD extends Objective {

  override def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int = numFeatures

  override def apply(theta: DenseVector[Double],
            batch: Int,
            batchLines: Seq[String],
            featureIndex: Index[String])
           (implicit config: LLLMParams,
            cache: ResultCache): (Double, DenseVector[Double]) = {

    var ll = 0d
    val grad = DenseVector.zeros[Double](theta.length)

    val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
    val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = cache.getDisk(Symbol(s"NoiseFeatures$batch"))

    batchDataSamples zip batchNoiseSamples foreach { case (data, noise) =>

      // here we're just using the "noise" samples from to get a monte carlo estimate of the partition

      // there doesn't seem to be a significant performance difference between the following two impls:

      // "fast"

      //              val fvData = new FeatureVector(data)
      //              val score = theta dot fvData
      //              ll += score
      //              axpy(1d, fvData, grad)
      //
      //              //val noiseExpScores = new ArrayBuffer[Double](noise.size + 1)
      //              val noiseExpScores = new Array[Double](noise.size + 1)
      //              var i = 0
      //              while (i < noise.size) {
      //                noiseExpScores.update(i, exp(theta dot new FeatureVector(noise(i))))
      //                i += 1
      //              }
      //              noiseExpScores.update(noise.size, exp(score))
      //              val norm = sum(noiseExpScores) / noiseExpScores.length * vocabIndex.size
      //              ll -= log(norm)
      //
      //              i = 0
      //              while (i < noise.size) {
      //                axpy(-noiseExpScores(i) / norm, new FeatureVector(noise(i)), grad)
      //                i += 1
      //              }
      //              axpy(-noiseExpScores(noise.size) / norm, fvData, grad)


      // "slow"

      val score = theta dot new FeatureVector(data)
      val noiseExpScores = (noise :+ data).map(x => exp(theta dot new FeatureVector(x))).toSeq
      val norm = sum(noiseExpScores) // just a constant factor to make denom the right size // / (1 + noise.length) * vocabIndex.size

      ll += score
      ll -= log(norm)
      //assert(score - log(norm) <= 0)

      axpy(1d, new FeatureVector(data), grad)
      noiseExpScores zip (noise :+ data) foreach { case (dScore, dFeat) => axpy(-dScore / norm, new FeatureVector(dFeat), grad)}
    }
    (ll, grad)
  }

}
