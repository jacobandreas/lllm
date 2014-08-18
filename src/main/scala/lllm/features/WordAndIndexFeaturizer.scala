package lllm.features

import breeze.linalg.SparseVector
import erector.learning.Feature

/**
 * @author jda
 */
case class WordAndIndexFeaturizer() extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Array[Feature] = {
    if (arg.length - 2 < 0) return new Array[String](0)
    val r = new Array[Feature](arg.length - 2)
    var i = 0
    while (i < arg.length - 2) {
      r(i) = s"SKIP__${arg(i)}_$i"
      i += 1
    }
    r
  }

}
