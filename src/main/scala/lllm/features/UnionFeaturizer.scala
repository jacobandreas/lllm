package lllm.features

/**
 * @author jda
 */
case class UnionFeaturizer(f1: Featurizer, f2: Featurizer) extends Featurizer {
  override def apply(arg: IndexedSeq[String]): Iterable[String] = f1(arg) ++ f2(arg)
}
