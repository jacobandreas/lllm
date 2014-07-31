package lllm.features

import breeze.linalg.Counter

/**
 * @author jda
 */
class FrequentSuffixPreprocessor(suffixCounts: Counter[String,Double],
                                 threshold: Double,
                                 unknownWordToken: String) extends WordPreprocessor with Serializable {

  def apply(arg: String): String = arg.tails.find(suffixCounts(_) >= threshold).getOrElse(unknownWordToken)

}

object FrequentSuffixPreprocessor {

  def apply(counts: Counter[String,Double], threshold: Double, unknownWordToken: String): FrequentSuffixPreprocessor = {
    val suffixCounts = Counter(counts.activeKeysIterator.flatMap { key =>
      key.tails.map(_ -> 1d)
    })
    new FrequentSuffixPreprocessor(suffixCounts, threshold, unknownWordToken)
  }

}
