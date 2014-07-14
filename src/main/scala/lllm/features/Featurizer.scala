package lllm.features

/**
 * @author jda
 */
trait Featurizer extends (IndexedSeq[String] => Iterable[String]) with Serializable {

  def +(other: Featurizer) = UnionFeaturizer(this, other)

}
