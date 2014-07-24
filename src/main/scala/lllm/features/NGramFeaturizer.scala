package lllm.features

/**
 * @author jda
 */
case class NGramFeaturizer(minOrder: Int, maxOrder: Int, includePred: Boolean = true) extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Iterable[String] = {
    if (includePred)
      (minOrder to maxOrder) map { arg.takeRight(_).mkString("__") }
    else
      (minOrder to maxOrder) map { len => arg.takeRight(len).take(len-1).mkString("__") }
  }

}