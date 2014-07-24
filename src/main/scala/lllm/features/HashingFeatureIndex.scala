package lllm.features

import breeze.util.Index

import scala.collection.{MapLike, mutable}
import scala.util.hashing.MurmurHash3

/**
 * @author jda
 */
case class HashingFeatureIndex(positiveIndex: Index[String],
                               nPositive: Int = -1,
                               nNegative: Int = -1) extends Index[String] {

  val realNPositive = if (nPositive == -1) positiveIndex.size else nPositive
  val realNNegative = if (nNegative == -1) realNPositive else nNegative

  val positiveHasher =
    if (realNPositive == positiveIndex.size)
      positiveIndex
    else
      (x: String) => MurmurHash3.mix(45691, x.##) % realNPositive

  val negativeHasher =
    (x: String) => realNPositive + MurmurHash3.mix(65827, x.##) % realNNegative

  override def apply(k: String) = {
    if (positiveIndex contains k)
      positiveHasher(k)
    else
      negativeHasher(k)
  }

  override def size = realNPositive + realNNegative

  override def unapply(i: Int): Option[String] = ???

  override def pairs: Iterator[(String, Int)] = ???

  override def iterator: Iterator[String] = ???
}

