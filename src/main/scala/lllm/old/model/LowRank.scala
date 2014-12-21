package lllm.old.model

import breeze.linalg.{sum, axpy, DenseMatrix, DenseVector}
import breeze.numerics._
import breeze.util.Index
import igor.experiment.ResultCache
import lllm.old.main.LLLMParams

/**
 * @author jda
 */
case object LowRank extends Objective {

  override def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int = {
    numFeatures * config.rank
  }

  override def apply(vecTheta: DenseVector[Double],
                     batch: Int,
                     batchLines: Seq[String],
                     featureIndex: Index[String])
                    (implicit config: LLLMParams,
                     cache: ResultCache): (Double, DenseVector[Double]) = {

    val theta = vecTheta.asDenseMatrix.reshape(featureIndex.size, config.rank)
    var ll = 0d
    val grad = DenseMatrix.zeros[Double](featureIndex.size, config.rank)

    val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
    val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = cache.getDisk(Symbol(s"NoiseFeatures$batch"))

    batchDataSamples zip batchNoiseSamples foreach { case (data, noise) =>

      // TODO(jda) every pair of for loops does twice as much work as necessary
      var score = 0d
      data.foreach { featI =>
        data.foreach { featJ =>
          //score += dots(featI, featJ)
          score += theta(featI,::).t dot theta(featJ,::).t
        }
      }
      val denomScores = (noise :+ data) map { sample: Array[Int] =>
        var sampleScore = 0d
        sample.foreach { featI =>
          sample.foreach { featJ =>
            //sampleScore += dots(featI, featJ)
            sampleScore += theta(featI,::).t dot theta(featJ,::).t
          }
        }
        exp(sampleScore)
      }

      val norm = sum(denomScores)

      ll += score
      ll -= log(norm)

      data.foreach { featI =>
        data.foreach { featJ =>
          axpy(2d, theta(featJ,::).t, grad(featI,::).t)
        }
      }
      denomScores zip (noise :+ data) foreach { case (dScore, dFeat) =>
        dFeat.foreach { featI =>
          dFeat.foreach { featJ =>
            axpy(-2d * dScore / norm, theta(featJ,::).t, grad(featI,::).t)
          }
        }
      }
    }
    (ll, grad.toDenseVector)
  }
}
