package lllm.features

import erector.learning.Feature

/**
 * @author jda
 */
case class NGramFeaturizer(minOrder: Int, maxOrder: Int) extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Array[Feature] = {
    val r = new Array[Feature](maxOrder - minOrder + 1)
    var i = minOrder
    //println(minOrder, maxOrder, arg.length)
    while (i <= maxOrder) {
      //println(arg.length - i, arg.length - 1)
      r(i - minOrder) = "NGRAM__" + arg.slice(arg.length - i, arg.length - 1).mkString("__")
      i += 1
    }
    r
  }

}