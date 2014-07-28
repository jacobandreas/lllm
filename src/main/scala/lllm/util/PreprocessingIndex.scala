package lllm.util

import breeze.util.Index

/**
 * @author jda
 */
case class PreprocessingIndex[T](preprocess: T => T)(source: Iterable[T]) extends Index[T] {

  val backingIndex = Index(source map (k => preprocess(k)))

  override def apply(t: T): Int = backingIndex(preprocess(t))

  override def unapply(i: Int): Option[T] = ???

  override def pairs: Iterator[(T, Int)] = backingIndex.pairs

  override def iterator: Iterator[T] = backingIndex.iterator
}
