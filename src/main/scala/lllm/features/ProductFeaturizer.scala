package lllm.features

import erector.learning.Feature

/**
 * @author jda
 */
case class ProductFeaturizer(f1: Featurizer, f2: Featurizer) extends Featurizer {
  override def apply(arg: IndexedSeq[String]): Array[Feature] = {
    val feats1 = f1(arg)
    val feats2 = f2(arg)
    val r = new Array[Feature](feats1.length * feats2.length)
    var i = 0
    while (i < feats1.length) {
      var j = 0
      while (j < feats2.length) {
        r(i * feats1.length + j) = s"PROD(${feats1(i)},${feats2(j)})"
        j += 1
      }
      i += 1
    }
    r
  }
}
