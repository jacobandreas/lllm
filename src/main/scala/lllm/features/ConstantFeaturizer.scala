package lllm.features

/**
 * @author jda
 */
case class ConstantFeaturizer() extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Iterable[String] = Seq("CONST")

}
