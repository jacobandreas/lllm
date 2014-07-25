package lllm.main

import igor.experiment.{ResultCache, Experiment, Stage}
import lllm.model.{LanguageModel, LogLinearLanguageModel}
import breeze.linalg.{softmax, DenseVector}
import erector.corpus.TextCorpusReader
import breeze.numerics.{exp, log2, log}

/**
 * @author jda
 */
object Evaluate extends Stage[LLLMParams] {

  override def run(config: LLLMParams, cache: ResultCache): Unit = {
    val model: LogLinearLanguageModel = cache.get('Model)
    //val model: LanguageModel = get('UnigramModel)
    val testCorpus = TextCorpusReader(config.testPath)
    //logger.info(model.theta.toString)
    val logProb = testCorpus.nGramIterator(config.order).foldLeft(0d) { (accum, ngram) => accum + model.logProb(ngram) }
    val logPP = logProb / testCorpus.nGramIterator(config.order).length
    logger.info(exp(-logPP).toString)
  }

}
