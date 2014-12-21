package lllm.model

/**
 * @author jda
 */
sealed trait Regularizer {
}

object Regularizer {
  case object L2 extends Regularizer
  case object L1 extends Regularizer
  case object KNL2 extends Regularizer
  case object Distributional extends Regularizer
}
