package lllm.model

import breeze.linalg.{sum, Counter}
import breeze.numerics.log
import lllm.features.WordPreprocessor

/**
 * @author jda
 */
class NGramLanguageModel(counts: Counter[IndexedSeq[String],Double],
                         contextPreprocessor: WordPreprocessor,
                         predictionPreprocessor: WordPreprocessor) extends LanguageModel with Serializable {

  // TODO(normalize)

  override def prob(ngram: IndexedSeq[String]): Double = {
    //val filtered = ngram map preprocessor
    val filtered = ngram.take(ngram.length - 1).map(contextPreprocessor) :+ predictionPreprocessor(ngram.last)
    if (counts contains filtered)
      counts(filtered)
    else
      .001
    // TODO(jda) this is horrible---at least do discounting
  }

  override def logProb(ngram: IndexedSeq[String]): Double = log(prob(ngram))

}
