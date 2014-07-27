package lllm.main

import breeze.collection.mutable.OpenAddressHashArray
import breeze.linalg.{CSCMatrix, DenseVector}
import breeze.util.Index
import lllm.util.Arrays

import scala.collection.mutable
import scala.collection.mutable.ArrayBuffer
import scala.util.hashing.MurmurHash3

/**
 *
 * @author dlwh
 */
@SerialVersionUID(1L)
class CrossProductIndex private (val firstIndex: Index[String],
                                 val secondIndex: Index[String],
                                 mapping: Array[OpenAddressHashArray[Int]],
                                 labelPartOfFeature: Array[Int],
                                 surfacePartOfFeature: Array[Int],
                                 id: String= "CrossProductIndex",
                                 val numHashBuckets: Int=0) extends Index[String] with Serializable {

  private val regex = (id + "\\((.*)::::(.*)\\)").r
  private val negHash = ("Hash" + "\\((.*)\\)").r

  def apply(t: String): Int = t match {
    case regex(a,b) =>
      mapped(firstIndex(a), secondIndex(b))
    case negHash(x)  if(x.toInt < numHashBuckets) =>
      x.toInt + trueSize
    case _ => -1
  }

  def mapped(labelFeature: Int, surfaceFeature: Int):Int = {
    if(labelFeature < 0 || surfaceFeature < 0) {
      -1
    } else {
      val arr = mapping(labelFeature)
      val f = if(arr ne null) {
        arr(surfaceFeature)
      } else {
        -1
      }

      if(f >= 0 || numHashBuckets == 0) {
        f
      } else {
        val hf = MurmurHash3.mixLast(MurmurHash3.mix(10891, labelFeature.##), surfaceFeature.##).abs
        (hf % numHashBuckets) + trueSize
      }
    }

  }


  private val trueSize = labelPartOfFeature.length
  override def size: Int = trueSize + numHashBuckets

  def unapply(i: Int): Option[String] = if(i >= size || i < 0)  None else Some(get(i))

  override def get(i: Int): String = {
    if (i >= size || i < 0) {
      throw new NoSuchElementException(s"index $i is not in CrossProductIndex of size $size")
    } else if (i < trueSize) {
      id + "(" + firstIndex.get(labelPartOfFeature(i)) + "::::" + secondIndex.get(surfacePartOfFeature(i)) + ")"
    } else {
      "Hash(" + (i - trueSize) + ")"
    }
  }

  def pairs: Iterator[(String, Int)] = Iterator.range(0,size).map(i => get(i) -> i)

  def iterator: Iterator[String] = Iterator.range(0,size).map(i => get(i))

  def crossProduct(lFeatures: Array[Int], sFeatures: Array[Int], offset: Int = 0, usePlainLabelFeatures: Boolean = true):Array[Int] = {
    val builder = new mutable.ArrayBuilder.ofInt
    builder.sizeHint(lFeatures.length * sFeatures.length)
    var i = 0
    while(i < lFeatures.length) {
      var j = 0
      while(j < sFeatures.length) {
        val m = mapped(lFeatures(i),sFeatures(j)) + offset
        if(m != -1)
          builder += m
        j += 1
      }

      i += 1
    }

    builder.result()
  }

  def buildSparseMatrix(weights: DenseVector[Double]):CSCMatrix[Double] = {
    val builder = new CSCMatrix.Builder[Double](firstIndex.size, secondIndex.size)

    if(numHashBuckets == 0) {
      // if no hash features, we can just iterate over the enumerated part of the index
      for(((l, s), i) <- (labelPartOfFeature zip surfacePartOfFeature).zipWithIndex) {
        val w = weights(i)
        if(w != 0.0)
          builder.add(l, s, w)
      }
    } else {
      // otherwise, check everything
      for(l <- 0 until firstIndex.size; s <- 0 until secondIndex.size) {
        val i = mapped(l, s)
        if(i >= 0 && weights(i) != 0) {
          builder.add(l, s, weights(i))
        }
      }
    }


    builder.result()
  }
}

object CrossProductIndex {

  class Builder(firstIndex: Index[String],
                      secondIndex: Index[String],
                      hashFeatures: Double = 1.0,
                      id: String = "CrossProductIndex") {
    def add(a: String, b: String):Int = add(firstIndex(a), secondIndex(b))

    private val mapping = Array.fill(firstIndex.size)(new OpenAddressHashArray[Int](secondIndex.size max 1, -1, 4))
    private val labelPart, surfacePart = new ArrayBuffer[Int]()

    def size = labelPart.size

    def add(firstArray: Array[Int], secondArray: Array[Int]):Array[Int] = {
      Arrays.crossProduct(firstArray, secondArray)(add)
    }

    def add(first: Int, secondArray: Array[Int]):Array[Int] = {
      secondArray.map(add(first, _))
    }

    def add(first: Int, second: Int):Int = {
      if(first < 0 || second < 0) {
        -1
      } else {
        val currentIndex: Int = mapping(first)(second)
        if(currentIndex == -1) {
          val x = size
          mapping(first)(second) = x
          labelPart += first
          surfacePart += second
          x
        } else {
          currentIndex
        }
      }
    }

    def result() = {
      new CrossProductIndex(firstIndex,
        secondIndex,
        mapping,
        labelPart.toArray, surfacePart.toArray,
        id,
        (hashFeatures * labelPart.size).toInt)
    }
  }

}

