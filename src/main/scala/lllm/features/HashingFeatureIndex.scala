package lllm.features

import breeze.util.Index
import erector.learning.Feature

import scala.collection.{MapLike, mutable}
import scala.util.hashing.MurmurHash3

/**
 * @author jda
 */
case class HashingFeatureIndex(positiveIndex: Index[Feature],
                               nPositive: Int = -1,
                               nNegative: Int = -1) extends Index[Feature] {

  val realNPositive = if (nPositive == -1) positiveIndex.size else nPositive
  val realNNegative = if (nNegative == -1) realNPositive else nNegative

  val positiveHasher =
    if (realNPositive == positiveIndex.size)
      positiveIndex
    else
      (x: Feature) => MurmurHash3.mix(45691, x.##) % realNPositive

  val negativeHasher =
    (x: Feature) => realNPositive + MurmurHash3.mix(65827, x.##) % realNNegative

  override def apply(k: Feature) = {
    if (positiveIndex contains k)
      positiveHasher(k)
    else
      negativeHasher(k)
  }

  override def size = realNPositive + realNNegative

  override def unapply(i: Int): Option[Feature] = ???

  override def pairs: Iterator[(Feature, Int)] = ???

  override def iterator: Iterator[Feature] = ???
}

