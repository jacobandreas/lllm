package lllm.features

import erector.learning.Feature

/**
 * @author jda
 */
trait Featurizer extends (IndexedSeq[String] => Array[Feature]) with Serializable {

  def +(other: Featurizer) = UnionFeaturizer(this, other)
  def *(other: Featurizer) = ProductFeaturizer(this, other)

}
