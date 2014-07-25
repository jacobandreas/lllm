package lllm.features

import breeze.linalg.SparseVector

/**
 * @author jda
 */
case class WordAndIndexFeaturizer() extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Iterable[String] = {
    arg.take(arg.length - 1).zipWithIndex.map { case (w, i) => s"SKIP__${w}_${i}"}
  }

}
