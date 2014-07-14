package lllm.model

import breeze.linalg.{sum, Counter}
import breeze.numerics.log

/**
 * @author jda
 */
class UnigramLanguageModel(counts: Counter[String,Double]) extends LanguageModel with Serializable {

  counts /= sum(counts)

  override def prob(ngram: IndexedSeq[String]): Double = counts(ngram(ngram.length - 1))

  override def logProb(ngram: IndexedSeq[String]): Double = log(prob(ngram))

}
