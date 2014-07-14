package lllm.model

/**
 * @author jda
 */
trait LanguageModel {

  def prob(ngram: IndexedSeq[String]): Double
  def logProb(ngram: IndexedSeq[String]): Double

}
