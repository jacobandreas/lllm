package lllm.main

import igor.experiment.Stage
import lllm.model.{LanguageModel, LogLinearLanguageModel}
import breeze.linalg.{softmax, DenseVector}
import erector.corpus.TextCorpusReader
import breeze.numerics.{exp, log2, log}

/**
 * @author jda
 */
class Evaluate(val order: Int = 3,
               val testPath: String) extends Stage {

  override def run(): Unit = {
    val model: LogLinearLanguageModel = get('Model)
    //val model: LanguageModel = get('UnigramModel)
    val testCorpus = TextCorpusReader(testPath)
    //logger.info(model.theta.toString)
    val logProb = testCorpus.nGramIterator(order).foldLeft(0d) { (accum, ngram) => accum + model.logProb(ngram) }
    val logPP = logProb / testCorpus.nGramIterator(order).length
    logger.info(exp(-logPP).toString)
  }

}
