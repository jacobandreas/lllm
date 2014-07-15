package lllm.main

import igor.experiment.Stage
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.optimize.{DiffFunction, GradientTester, BatchDiffFunction}
import breeze.linalg._
import breeze.util.Index
import breeze.features.FeatureVector
import breeze.numerics.{log, exp}
import breeze.optimize.FirstOrderMinimizer.OptParams
import lllm.model.LogLinearLanguageModel
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable

/**
 * @author jda
 */
class TrainModel(foo: Int = 1) extends Stage {

  override def run(): Unit = {

    val featureIndex: Index[String] = getDisk[Index[String]]('FeatureIndex)

    val optimization = OptParams(useStochastic = true, batchSize = 5, maxIterations = 50)
    val optTheta = optimization.minimize(makeObjectiveCD, DenseVector.zeros[Double](featureIndex.size))
    //GradientTester.test(makeObjective, DenseVector.zeros[Double](featureIndex.size), toString = featureIndex.get)
    put('Model, new LogLinearLanguageModel(get('Featurizer), get('FeatureIndex), get('VocabIndex), optTheta))

  }

  def makeObjectiveCD: BatchDiffFunction[DenseVector[Double]] = {

    val featureIndex: Index[String] = getDisk[Index[String]]('FeatureIndex)
    val vocabIndex: Index[String] = getDisk[Index[String]]('VocabIndex)

    new BatchDiffFunction[DenseVector[Double]] {

      override def fullRange: IndexedSeq[Int] = 0 until get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {

        task("calculating") {

          val grad = DenseVector.zeros[Double](featureIndex.size)
          var ll = 0d

          logger.info(s"batch: $batch")

          batch.foreach { batchIndex =>
            val batchDataSamples: Seq[Array[Int]] = getDisk(Symbol(s"DataGroup$batchIndex"))
            val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = getDisk(Symbol(s"NoiseGroup$batchIndex"))

            batchDataSamples zip batchNoiseSamples foreach { case (data, noise) =>

              // here we're just using the "noise" samples from NCE to get a monte carlo estimate of the partition
              // this is wrong because the noise samples are non-uniform---just a quick hack until we add NCE

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
              val norm = sum(noiseExpScores) / (1 + noise.length) * vocabIndex.size

              ll += score
              ll -= log(norm)
              //assert(score - log(norm) <= 0)

              axpy(1d, new FeatureVector(data), grad)
              noiseExpScores zip (noise :+ data) foreach { case (dScore, dFeat) => axpy(-dScore / norm, new FeatureVector(dFeat), grad) }
            }
          }
          (-ll, -grad)
        }
      }
    }
  }
}
