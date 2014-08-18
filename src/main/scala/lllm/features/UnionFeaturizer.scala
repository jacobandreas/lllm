package lllm.features

import erector.learning.Feature

/**
 * @author jda
 */
case class UnionFeaturizer(f1: Featurizer, f2: Featurizer) extends Featurizer {
  override def apply(arg: IndexedSeq[String]): Array[Feature] = f1(arg) ++ f2(arg)
}
