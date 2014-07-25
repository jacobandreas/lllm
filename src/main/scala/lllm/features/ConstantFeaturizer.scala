package lllm.features

/**
 * @author jda
 */
case class ConstantFeaturizer(includePred: Boolean) extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Iterable[String] =
    if (includePred)
      Seq(s"${arg.last}__CONST")
    else
      Seq("__CONST")

}
