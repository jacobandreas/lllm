package lllm.model

import breeze.linalg.{sum, Counter}
import breeze.numerics.log

/**
 * @author jda
 */
class NGramLanguageModel(counts: Counter[IndexedSeq[String],Double],
                         preprocessor: String => String = x => x) extends LanguageModel with Serializable {

  counts /= sum(counts)

  val defaultProb = counts(IndexedSeq.fill(1)("fszggggbt"))

  override def prob(ngram: IndexedSeq[String]): Double = {
    val filtered = ngram map preprocessor
    if (counts contains filtered)
      counts(filtered)
    else
      .001
    // TODO(jda) this is horrible---at least do discounting
  }

  override def logProb(ngram: IndexedSeq[String]): Double = log(prob(ngram))

}
