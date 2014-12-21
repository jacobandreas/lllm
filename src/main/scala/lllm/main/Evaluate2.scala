package lllm.main

import breeze.util.Index
import igor.experiment.{ResultCache, Experiment, Stage}
import lllm.evaluation.PerplexityEvaluator
import lllm.model.{KenLMPerplexityEvaluator, HierarchicalLanguageModel, KneserNeyLanguageModel, LanguageModel}
import breeze.linalg.{softmax, DenseVector}
import erector.corpus.TextCorpusReader
import breeze.numerics.{exp, log2, log, Inf}

import scala.io.Source

/**
 * @author jda
 */
object Evaluate2 extends Stage[LLLMParams2] {

  override def run(implicit config: LLLMParams2, cache: ResultCache): Unit = {
    val lllmModel: HierarchicalLanguageModel = cache.get('Model)
    val models = Map("lllm" -> lllmModel)
    models.foreach { case (name, model) =>
      logger.info(name)
      val trainLogPP = PerplexityEvaluator(model, TextCorpusReader(config.trainPath).prefix(config.nLines))
      val testLogPP = PerplexityEvaluator(model, TextCorpusReader(config.testPath))
      logger.info(trainLogPP.toString)
      logger.info(testLogPP.toString)
    }
    dumpWeights(lllmModel)

    //val kenLogPP = KenLMPerplexityEvaluator("/Users/jda/Code/288/p1/assign1_data/test.en")
//    val kenLogPP = KenLMPerplexityEvaluator("/Users/jda/Corpora/lm/wsj/dev.txt")
//    println(kenLogPP)
  }

  def dumpWeights(model: HierarchicalLanguageModel)(implicit cache: ResultCache): Unit = {
    cache.writeFile('Weights,
      (0 until model.theta.length).toTraversable.map { i =>
        s"${model.featureIndex.get(i)} ${model.theta(i)}"
      }
    )
  }

}
