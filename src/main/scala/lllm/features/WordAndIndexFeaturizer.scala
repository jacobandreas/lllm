package lllm.features

import breeze.linalg.SparseVector

/**
 * @author jda
 */
case class WordAndIndexFeaturizer(includePred: Boolean = true) extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Iterable[String] = {
    //arg.zipWithIndex.map { case (w, i) => s"${w}__$i" }
    val last = if (includePred) arg.last else ""
    arg.take(arg.length - 1).zipWithIndex.map { case (w, i) => s"${w}__${i}__$last"}
  }

}
