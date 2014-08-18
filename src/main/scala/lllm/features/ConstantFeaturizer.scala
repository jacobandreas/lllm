package lllm.features

import erector.learning.Feature

/**
 * @author jda
 */
case class ConstantFeaturizer() extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Array[Feature] = Array("CONST")

}
