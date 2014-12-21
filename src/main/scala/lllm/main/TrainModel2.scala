package lllm.main

import breeze.linalg.{sum, DenseVector}
import breeze.numerics.{abs, exp}
import breeze.optimize.{GradientTester, BatchDiffFunction}
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.util.Implicits._
import erector.corpus.{TextCorpusReader2, TextCorpusReader}
import igor.experiment.{ResultCache, Stage}
import lllm.evaluation.PerplexityEvaluator
import lllm.model.{HierarchicalLanguageModel, Regularizer, HierarchicalObjective2}

/**
 * @author jda
 */
object TrainModel2 extends Stage[LLLMParams2] {

  override def run(implicit config: LLLMParams2, cache: ResultCache): Unit = {

    val objectiveComputer = HierarchicalObjective2
    val productIndex: CrossProductIndex = cache.getDisk('ProductIndex)
    val initTheta = DenseVector.zeros[Double](productIndex.size)

    val objective = new BatchDiffFunction[DenseVector[Double]] {
      override def fullRange: IndexedSeq[Int] = 0 until cache.get('NLineGroups)
      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {
        val setBatch = batch.toSet
        var ll = 0d
        val grad = DenseVector.zeros[Double](theta.length)
        val lines = TextCorpusReader(config.trainPath).prefix(config.nLines)
        //val lines = TextCorpusReader2(config.trainPath)
        lines.lineGroupIterator(config.lineGroupSize).zipWithIndex.filter(p => setBatch.contains(p._2)).foreach { case (batchLines, batchId) =>
          val (bll, bgrad) = objectiveComputer(theta, batchLines)
          ll += bll
          grad += bgrad
        }
        (-ll, -grad)
      }
    }

    val optimization = OptParams(useStochastic = true,
                                 batchSize = 256,
                                 //maxIterations = 30,
                                 maxIterations = cache.get[Int]('NLineGroups) * config.nEpochs,
                                 useL1 = config.regularizerType == Regularizer.L1,
                                 regularization = config.regularizerStrength)

//    val optimization = OptParams(useStochastic = false, maxIterations = 100, regularization = 0.5)

//    val optTheta = optimization.minimize(objective, initTheta)

//    val optModel = optimization.iterations(objective, initTheta).toIterable.map { optState =>
//      val model = new HierarchicalLanguageModel(cache.get('ContextFeaturizer),
//                                                cache.get('ProductIndex),
//                                                cache.get('VocabularyIndex),
//                                                cache.get('HuffmanDict),
//                                                optState.x)
//      val trainScore = PerplexityEvaluator(model, TextCorpusReader(config.trainPath).prefix(1000))
//      val testScore = PerplexityEvaluator(model, TextCorpusReader(config.testPath))
//      //if (optState.iter % cache.get[Int]('NLineGroups) == 0) {
//      if (optState.iter % 20 == 0) {
//        logger.info(s"train: $trainScore, test: $testScore")
//      }
//      model
//    }.last

    def buildModel(theta: DenseVector[Double]) = new HierarchicalLanguageModel(cache.get('ContextFeaturizer),
                                                                               cache.get('ProductIndex),
                                                                               cache.get('VocabularyIndex),
                                                                               cache.get('HuffmanDict),
                                                                               theta)

    val optTheta = optimization.iterations(objective, initTheta).take(optimization.maxIterations).tee { optState =>
      if (optState.iter % 10 == 0) {
        val model = buildModel(optState.x)
        val trainScore = PerplexityEvaluator(model, TextCorpusReader(config.trainPath).prefix(1000))
        val testScore = PerplexityEvaluator(model, TextCorpusReader(config.testPath))
        logger.info(s"train: $trainScore, test: $testScore")
      }
//      if (optState.iter != 0 && optState.iter % 50 == 0) {
//        val model = buildModel(optState.x)
//        cache.putDisk(Symbol(s"Model${optState.iter / 50}"), model)
//      }
    }.last.x

    val optModel = buildModel(optTheta)

    //GradientTester.test(objective, optTheta, toString = (x: Int) => x.toString)

    cache.put('Model, optModel)
  }

}
