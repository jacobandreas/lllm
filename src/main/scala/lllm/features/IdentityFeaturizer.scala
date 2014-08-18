package lllm.features

import breeze.util.Index
import erector.learning.Feature

/**
 * @author jda
 */
case class IdentityFeaturizer() extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Array[Feature] = Array(arg.last)

}
