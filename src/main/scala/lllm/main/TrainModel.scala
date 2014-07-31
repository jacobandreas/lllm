package lllm.main

import breeze.features.FeatureVector
import breeze.linalg._
import breeze.numerics.{exp, log}
import breeze.optimize.{GradientTester, BatchDiffFunction}
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.util.Index
import igor.experiment.{ResultCache, Stage}
import lllm.model._

/**
 * @author jda
 */
object TrainModel extends Stage[LLLMParams] {

  override def run(config: LLLMParams, cache: ResultCache): Unit = {

    implicit val _c = config
    implicit val _r = cache

    val featureIndex: CrossProductIndex = cache.getDisk('CrossIndex)

    // TODO(jda) low rank is actually orthogonal to everything else---generalize later
    val initTheta = DenseVector.zeros[Double](config.objective.numParams(featureIndex.size))

    val optParams = OptParams(useStochastic = true, batchSize = 5, maxIterations = 100, regularization = 0.1)
    val objective = makeObjective(config.objective)
    //GradientTester.test(objective, initTheta, toString = (x: Int) => x.toString)
    val optTheta = optParams.minimize(objective, initTheta)
    cache.put('Model, new LogLinearLanguageModel(cache.get('PredictionFeaturizer),
      cache.get('ContextFeaturizer),
      cache.getDisk('CrossIndex),
      optTheta))

  }

  def makeObjective(objective: Objective)(implicit config: LLLMParams, cache: ResultCache): BatchDiffFunction[DenseVector[Double]] = {
    val featureIndex: Index[String] = cache.getDisk('CrossIndex)
    new BatchDiffFunction[DenseVector[Double]] {
      override def fullRange: IndexedSeq[Int] = 0 until cache.get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batchGroups: IndexedSeq[Int]): (Double, DenseVector[Double]) = {
        var ll = 0d
        val grad = DenseVector.zeros[Double](theta.length)
        batchGroups.foreach { batch =>
          val (bll, bgrad) = objective(theta, batch, featureIndex)
          ll += bll
          grad += bgrad
        }
        (-ll, -grad)
      }
    }
  }
}
