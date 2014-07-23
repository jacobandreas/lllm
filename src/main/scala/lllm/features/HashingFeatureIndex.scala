package lllm.features

import breeze.util.Index

import scala.collection.mutable
import scala.util.hashing.MurmurHash3

/**
 * @author jda
 */
case class HashingFeatureIndex(positiveIndex: Index[String],
                               nPositive: Int = positiveIndex.size,
                               nNegative: Int = nPositive) extends (String => Int) {

  val positiveHasher =
    if (nPositive == positiveIndex.size)
      positiveIndex
    else
      (x: String) => MurmurHash3.mix(45691, x.##) % nPositive

  val negativeHasher =
    (x: String) => nPositive + MurmurHash3.mix(65827, x.##) % nNegative

  override def apply(k: String) = {
    if (positiveIndex contains k)
      positiveHasher(k)
    else
      negativeHasher(k)
  }

  def size = nPositive + nNegative

}

