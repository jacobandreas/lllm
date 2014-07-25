package lllm.features

/**
 * @author jda
 */
case class IdentityFeaturizer() extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Iterable[String] = Seq(arg.last)

}
