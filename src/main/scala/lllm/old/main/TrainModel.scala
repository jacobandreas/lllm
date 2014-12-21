package lllm.old.main

import breeze.linalg._
import breeze.optimize.BatchDiffFunction
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.util.Index
import erector.corpus.TextCorpusReader
import igor.experiment.{ResultCache, Stage}
import lllm.main.CrossProductIndex
import lllm.model._
import lllm.old.model.{LogLinearLanguageModel, Objective, Hierarchical}

import scala.io.Source

/**
 * @author jda
 */
object TrainModel extends Stage[LLLMParams] {

  override def run(implicit config: LLLMParams, cache: ResultCache): Unit = {

//    implicit val _c = config
//    implicit val _r = cache

    val featureIndex: CrossProductIndex = cache.getDisk('CrossIndex)

    val initTheta = DenseVector.zeros[Double](config.objective.numParams(featureIndex.size))

    val optParams = OptParams(useStochastic = true, batchSize = 50, maxIterations = 200, regularization = 1)
    val objective = makeObjective(config.objective)
    //GradientTester.test(objective, initTheta, toString = (x: Int) => x.toString)
    val optTheta = optParams.minimize(objective, initTheta)
    val model =
      if (config.objective == Hierarchical) {
        new HierarchicalLanguageModel(cache.get('ContextFeaturizer),
                                      cache.get('CrossIndex),
                                      cache.get('VocabularyIndex),
                                      cache.get('HuffmanDict),
                                      optTheta)
      } else {
        new LogLinearLanguageModel(cache.get('PredictionFeaturizer),
                                   cache.get('ContextFeaturizer),
                                   cache.get('CrossIndex),
                                   optTheta)
      }
    cache.put('Model, model)

  }

  def makeObjective(objective: Objective)(implicit config: LLLMParams, cache: ResultCache): BatchDiffFunction[DenseVector[Double]] = {
    val featureIndex: Index[String] = cache.getDisk('CrossIndex)
    new BatchDiffFunction[DenseVector[Double]] {
      override def fullRange: IndexedSeq[Int] = 0 until cache.get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {
        val setBatch = batch.toSet
        var ll = 0d
        val grad = DenseVector.zeros[Double](theta.length)
        val lines = new TextCorpusReader(Source.fromFile(config.trainPath))
        lines.lineGroupIterator(config.featureGroupSize).zipWithIndex.filter(p => setBatch.contains(p._2)).foreach { case (batchLines, batchId) =>
//          println(batchId)
//          println(batchLines.mkString("\n"))
          val (bll, bgrad) = objective(theta, batchId, batchLines, featureIndex)
            ll += bll
            grad += bgrad
        }
//        batchLines.foreach { batch =>
//          val (bll, bgrad) = objective(theta, batch, featureIndex)
//          ll += bll
//          grad += bgrad
//        }
        (-ll, -grad)
      }
    }
  }
}
