package lllm.main

import breeze.util.Index
import igor.experiment.{ResultCache, Experiment, Stage}
import lllm.model.{KneserNeyLanguageModel, LanguageModel, LogLinearLanguageModel}
import breeze.linalg.{softmax, DenseVector}
import erector.corpus.TextCorpusReader
import breeze.numerics.{exp, log2, log, Inf}

/**
 * @author jda
 */
object Evaluate extends Stage[LLLMParams] {

  override def run(config: LLLMParams, cache: ResultCache): Unit = {
    implicit val (_c, _r) = (config, cache)
    val lllmModel: LanguageModel = cache.get('Model)
    val baselineModel: LanguageModel = cache.get('NGramModel)

    val models = Map("baseline" -> baselineModel, "lllm" -> lllmModel)
    //val models = Map("baseline" -> baselineModel)

    models.foreach { case (name, model) =>
      logger.info(name)
      val trainLogPP = computePerplexity(model, TextCorpusReader(config.trainPath).dummy(config.nLines))
      val testLogPP = computePerplexity(model, TextCorpusReader(config.testPath))
      logger.info(exp(trainLogPP).toString)
      logger.info(exp(testLogPP).toString)
    }
  }

  def computePerplexity(model: LanguageModel, corpus: TextCorpusReader)(implicit config: LLLMParams, cache: ResultCache): Double = {
    var logProb = 0d
    var count = 0d
    val iter = corpus.nGramIterator(config.order)
    while (iter.hasNext) {
      val ngram = iter.next()
      val score = model.logProb(ngram)

      if (score > -Inf) {
        // TODO(jda) check to make sure the word is actually OOV
//        println(exp(score))
//        println(ngram)
        logProb += score
        count += 1

//        val vocab: Index[String] = cache.get('VocabIndex)
//        var total = 0d
//        vocab.foreach { word =>
//          val altNGram = ngram.dropRight(1) :+ word
//          val sc = model.prob(altNGram)
//          //println(altNGram, sc)
//          total += sc
//        }
//        println(total)
//        if (total > 1) {
//          model.asInstanceOf[KneserNeyLanguageModel].prob(ngram, report = true)
//          System.exit(1)
//        }
      }
    }
    //val logPP = logProb / testCorpus.nGramIterator(config.order).length
    - logProb / count
  }

}
