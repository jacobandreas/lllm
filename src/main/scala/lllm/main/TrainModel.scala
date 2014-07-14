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

  def myDot(theta: DenseVector[Double], phi: Array[Int]): Double = {
    phi.foldLeft(0d)((accum, i) => accum + theta(i))
  }

  def myAxpy(a: Double, x: Array[Int], y: Vector[Double]) {
    x.foreach { y(_) += a }
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

          batchDataSamples zip batchNoiseSamples foreach { case (data, noise) =>

            // here we're just using the "noise" samples from NCE to get a monte carlo estimate of the partition
            // this is wrong because the noise samples are non-uniform---just a quick hack until we add NCE
            // myX methods are faster than instantiating a new FeatureVector
            // (TODO: just make the FeatureVector once when we load the data)

            val score = myDot(theta, data)

            val noiseExpScores = (noise :+ data).map(n => exp(myDot(theta,n))).toSeq

            val norm = sum(noiseExpScores) / (1 + noise.length) * vocabIndex.size

            ll += score
            ll -= log(norm)

            //assert(score - log(norm) <= 0)

            myAxpy(1d, data, grad)
            noiseExpScores zip (noise :+ data) foreach { case (sc, ft) => myAxpy(-sc / norm, ft, grad) }
          }

        }

        (-ll, -grad)

        }

      }

    }
  }

}
