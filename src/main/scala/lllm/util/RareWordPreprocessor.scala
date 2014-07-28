package lllm.util

import breeze.linalg.Counter

/**
 * @author jda
 */
case class RareWordPreprocessor(counts: Counter[String,Double], threshold: Double, oovStr: String) extends (String => String) {

  val known = Set(counts.toMap.flatMap { case (word, count) =>
    if (count <= threshold)
      None
    else
      Some(word)
  }.toSeq:_*)

  def apply(arg: String): String =
    if (known contains arg)
      arg
    else
      oovStr

}
