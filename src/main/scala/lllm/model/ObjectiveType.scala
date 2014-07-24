package lllm.model

/**
 * @author jda
 */
object ObjectiveType extends Enumeration {
  type ObjectiveType = Value
  val Exact, CD, NCE, Hierarchical, LowRank = Value
}

