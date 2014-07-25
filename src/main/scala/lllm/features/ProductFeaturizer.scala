package lllm.features

/**
 * @author jda
 */
case class ProductFeaturizer(f1: Featurizer, f2: Featurizer) extends Featurizer {
  override def apply(arg: IndexedSeq[String]): Iterable[String] =
    for {
      s1 <- f1(arg)
      s2 <- f2(arg)
    } yield s"PROD($s1,$s2)"
}
