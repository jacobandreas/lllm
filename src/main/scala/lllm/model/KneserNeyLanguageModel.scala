package lllm.model

import breeze.linalg.{sum, max, Counter}
import breeze.numerics.log
import lllm.features.WordPreprocessor

/**
 * @author jda
 */
//class KneserNeyLanguageModel(disCounts: Map[Int,Counter[IndexedSeq[String],Double]],
//                             continuations: Counter[String,Double],
//                             interpolations: Map[Int,Double],
//                             contextPreprocessor: WordPreprocessor,
//                             predictionPreprocessor: WordPreprocessor) extends LanguageModel {
class KneserNeyLanguageModel(tokenCounts: Map[IndexedSeq[String],Double],
                             tokenCountNormalizers: Map[IndexedSeq[String],Double],
                             typeContinuationCounts: Map[Int,Map[IndexedSeq[String],Double]],
                             typeContinuationCountNormalizers: Map[Int,Map[IndexedSeq[String],Double]],
                             typeContextCounts: Map[Int,Map[IndexedSeq[String],Double]],
                             vocabulary: Set[String],
                             discount: Double) extends LanguageModel with Serializable {

//  override def prob(ngram: IndexedSeq[String]): Double = {
//    var prob = 0d
//    var i = ngram.length
//    var scale = 1
//    while (i >= 1) {
//      prob += scale * disCounts(i)(ngram.takeRight(i))
//      scale *= interpolations(i-1)
//      i += 1
//    }
//    prob += scale * continuations(ngram.last)
//    prob
//  }

  override def prob(ngram: IndexedSeq[String]): Double = prob(ngram, false)

  def prob(ngram: IndexedSeq[String], report: Boolean): Double = {

//    if (report) {
//      println(tokenCounts contains ngram.dropRight(1))
//    }

    var prob = 0d
    var scale = 1d
    val context = ngram.dropRight(1)
//    println(tokenCounts)
    if (tokenCounts contains ngram) {
//      println("found ngram " + ngram + " in tokenCounts")
      prob += (tokenCounts(ngram) - discount) / tokenCountNormalizers(context)
//      println(">>", ngram, context, tokenCounts(ngram), tokenCountNormalizers(context))

      if (report) {
        println("HIGHEST ORDER")
        println(ngram)
        println(tokenCounts)
        println(tokenCountNormalizers)
        println(tokenCounts(ngram))
        println(tokenCountNormalizers(context))
        println(typeContextCounts(ngram.length - 1))
        println(prob)
        //      System.exit(1)
      }
    }

    if (tokenCountNormalizers.contains(context)) {
      scale *= discount * typeContextCounts(ngram.length - 1)(context) / tokenCountNormalizers(context)
    }

    var history = ngram.length - 1
    while (history > 0) {
      val cutNGram = ngram.takeRight(history)
      val cutContext = cutNGram.dropRight(1)
//      println("history " + history)
//      println("back off to " + cutNGram)
//      println(typeContinuationCounts(history))

//      if (report) {
//        println(typeContinuationCountNormalizers(history).contains(cutNGram.dropRight(1)))
//      }

      if (typeContinuationCounts(history) contains cutNGram) {
        prob += scale * (typeContinuationCounts(history)(cutNGram) - discount) / typeContinuationCountNormalizers(history)(cutContext)
        if (history > 1) {
          //println(typeContextCounts(history-1))
          //println(typeContextCounts(history)(cutContext))
          //println(tokenCountNormalizers(cutContext))
        }

//        if (report) {
//          println("BACKOFF")
//          println(history)
//          println(cutNGram)
//          println(typeContinuationCounts(history))
//          println(typeContinuationCountNormalizers(history))
//          //        System.exit(1)
//          println("\n\n")
//        }
      }

      if (typeContinuationCountNormalizers(history).contains(cutContext)) {
        scale *= discount * typeContextCounts(history-1)(cutContext) / typeContinuationCountNormalizers(history)(cutContext)
      }
      history -= 1
    }

//    if (prob == 0) {
//      assert(!vocabulary.contains(ngram.last))
////      println("can't predict " + ngram.last)
//    }
//    else {
////      println(ngram)
////      println(prob)
////      println(typeContinuationCountNormalizers(2))
//    }
//    println()

    prob
  }

  override def logProb(ngram: IndexedSeq[String]): Double = {
    log(prob(ngram))
  }

}

object KneserNeyLanguageModel {

  def apply(counts: Map[Int,Counter[IndexedSeq[String],Double]],
            contextPreprocessor: WordPreprocessor,
            predictionPreprocessor: WordPreprocessor): KneserNeyLanguageModel = {

    val maxOrder = max(counts.keys)

    //    def discount(c: Int): Double = c match {
    //      case 0 => 0
    //      case 1 => 0.1
    //      case 2 => 0.1
    //      case _ => 0.1
    //    }

    val tokenCounts = counts(maxOrder)
    val tokenCountNormalizers = Counter(tokenCounts.iterator.map { case (ngram, count) =>
      ngram.dropRight(1) -> count
    })

    val vocabulary = counts(1).keySet.map(_.head).toSet

    val typeContinuationCounts = counts.map { case (order, orderCounts) =>
      (order - 1) -> orderCounts.keySet.groupBy(_.tail).map { case (tail, ngrams) =>
//        if (order == 2) {
//          println(tail, ngrams)
//          println(ngrams.map(_.head))
//        }
        tail -> ngrams.map(_.head).size.toDouble
      }
    }
    val typeContinuationCountNormalizers = counts.map { case (order, orderCounts) =>
      (order - 1) -> orderCounts.keySet.groupBy(_.drop(1).dropRight(1)).map { case (mid, ngrams) =>
        //mid -> Set(ngrams.map(g => (g.head, g.last))).size.toDouble
        mid -> ngrams.map(g => (g.head, g.last)).size.toDouble
      }
    }

    val typeContextCounts = counts.map { case (order, orderCounts) =>
      (order - 1) -> orderCounts.keySet.groupBy(_.dropRight(1)).map { case (pref, ngrams) =>
        //pref -> Set(ngrams.map(_.last)).size.toDouble
        pref -> ngrams.map(_.last).size.toDouble
      }
    }

    new KneserNeyLanguageModel(tokenCounts.toMap,
                               tokenCountNormalizers.toMap,
                               typeContinuationCounts,
                               typeContinuationCountNormalizers,
                               typeContextCounts,
                               vocabulary,
                               0.1)
  }

}
