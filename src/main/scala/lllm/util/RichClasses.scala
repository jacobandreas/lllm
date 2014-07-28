package lllm.util

import breeze.linalg.Counter
import breeze.util.Index

/**
 * @author jda
 */
object RichClasses {

  // TODO(jda) this is not the right way to do this---cleaner to preprocess before indexing
  implicit class IndexWithOOV[T](index: Index[T]) {
    def collapseRare(threshold: Double, counts: Counter[T,Double], newKey: T): Index[T] = {
      Index {
        index map { key =>
            if (counts(key) <= threshold)
              newKey
            else
              key
        }
      }
    }
  }

  implicit class IndexWithSuffix(index: Index[String]) {
    def toFrequentSuffix = ???
  }

}
