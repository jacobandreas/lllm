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

/**
 * @author jda
 */
class TrainModel(foo: Int = 1) extends Stage {

  override def run(): Unit = {

    val featureIndex: Index[String] = getDisk[Index[String]]('FeatureIndex)

    val optimization = OptParams(useStochastic = true, batchSize = 5, maxIterations = 50)
    val optTheta = optimization.minimize(makeObjective, DenseVector.zeros[Double](featureIndex.size))
    //GradientTester.test(makeObjective, DenseVector.zeros[Double](featureIndex.size), toString = featureIndex.get)
    put('Model, new LogLinearLanguageModel(get('Featurizer), get('FeatureIndex), get('VocabIndex), optTheta))

  }

  def makeObjective: BatchDiffFunction[DenseVector[Double]] = {

    val featureIndex: Index[String] = getDisk[Index[String]]('FeatureIndex)
    val vocabIndex: Index[String] = getDisk[Index[String]]('VocabIndex)

    new BatchDiffFunction[DenseVector[Double]] {

      override def fullRange: IndexedSeq[Int] = 0 until get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {

        task("calculating") {

          val grad = DenseVector.zeros[Double](featureIndex.size)
          var ll = 0d

          logger.info(s"batch: $batch")

          batch.par.foreach { batchIndex =>
            val batchDataSamples: Seq[Array[Int]] = getDisk(Symbol(s"DataGroup$batchIndex"))
            val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = getDisk(Symbol(s"NoiseGroup$batchIndex"))

            batchDataSamples zip batchNoiseSamples foreach { case (rawData, rawNoise) =>

              // TODO(jda) is this cheaper than inline?
              val data = new FeatureVector(rawData)
              val noise = rawNoise map { new FeatureVector(_) }

              // here we're just using the "noise" samples from NCE to get a monte carlo estimate of the partition
              // this is wrong because the noise samples are non-uniform---just a quick hack until we add NCE

              // TODO(jda) remove maps and foreaches (or just go straight to the real training procedure?)
              val score = theta dot data
              val noiseExpScores = (noise :+ data).map(x => exp(theta dot x)).toSeq
              val norm = sum(noiseExpScores) / (1 + noise.length) * vocabIndex.size

              ll += score
              ll -= log(norm)
              //assert(score - log(norm) <= 0)

              axpy(1d, data, grad)
              noiseExpScores zip (noise :+ data) foreach { case (sc, ft) => axpy(-sc / norm, ft, grad) }
            }
          }
          (-ll, -grad)
        }
      }
    }
  }
}
