package lllm.evaluation

import breeze.util.Index
import breeze.numerics.{log, pow, abs, Inf}
import erector.corpus.TextCorpusReader
import igor.experiment.ResultCache
import lllm.main.LLLMParams2
import lllm.model.LanguageModel

/**
 * @author jda
 */
object PerplexityEvaluator {

  def apply(model: LanguageModel, corpus: TextCorpusReader)
           (implicit config: LLLMParams2, cache: ResultCache): Double = {
    val vocab: Index[String] = cache.get('VocabularyIndex)
    var logProb = 0d
    var count = 0d
    val iter = corpus.nGramIterator(config.order)
    val log2 = log(2d)
    while (iter.hasNext) {
      val ngram = iter.next()
      val score = model.logProb(ngram) / log2
      if (score > -Inf) {
        logProb += score
        count += 1
      }
      if (config.safeEval) {
        if (score == -Inf) {
          println(ngram.last)
          assert(!(vocab contains ngram.last))
        } else {
          var total = 0d
          vocab.foreach { word =>
            val altNGram = ngram.dropRight(1) :+ word
            val sc = model.prob(altNGram)
            total += sc
          }
          assert(abs(total) < 1e-6)
        }
      }
    }
    pow(2d, -logProb / count)
  }


}
