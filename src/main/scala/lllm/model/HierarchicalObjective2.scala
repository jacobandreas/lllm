package lllm.model

import breeze.features.FeatureVector
import breeze.linalg.{DenseVector,axpy}
import breeze.numerics._
import breeze.util.Index
import erector.corpus.TextCorpusReader
import erector.learning.Feature
import igor.experiment.ResultCache
import lllm.features.Featurizer
import lllm.main.{LLLMParams2, CrossProductIndex}
import lllm.util.HuffmanDict
import erector.util.text.toNGramIterable

/**
 * @author jda
 */
case object HierarchicalObjective2 {

  def apply(theta: DenseVector[Double], lines: Seq[String])
           (implicit config: LLLMParams2, cache: ResultCache): (Double, DenseVector[Double]) = {

    var ll = 0d
    val grad = DenseVector.zeros[Double](theta.length)

    val huffmanDict: HuffmanDict[Int] = cache.get('HuffmanDict)
    val productIndex: CrossProductIndex = cache.get('ProductIndex)
    val contextFeaturizer: Featurizer = cache.get('ContextFeaturizer)
    val contextFeatureIndex: Index[Feature] = cache.get('ContextFeatureIndex)
    val vocabIndex: Index[String] = cache.get('VocabularyIndex)

    lines.foreach { line =>
      // TODO(jda) s/start/stop
      val ngrams = (line.split(" ") :+ erector.util.text.DefaultStartSymbol).toIndexedSeq.nGrams(config.order)
      ngrams.foreach { ngram =>
        val wordId = vocabIndex(ngram.last)
        val contextIds = contextFeaturizer(ngram).map(contextFeatureIndex)
        val optCode = huffmanDict.dict.get(wordId)
        if (optCode.isDefined) {
          val code = optCode.get
          code.tails.toArray.filter(_.nonEmpty).foreach { codePrefix =>
            val decision = if (codePrefix.head == '1') 1d else -1d
            val history = codePrefix.tail
            val decisionIds = Array(huffmanDict.prefixIndex(history), huffmanDict.constIndex)
            val feats = productIndex.crossProduct(decisionIds, contextIds)
            val score = (theta dot new FeatureVector(feats)) * decision
//            println(exp(score))
//            println(log1p(exp(score)))
            ll += -log1p(exp(score))
            axpy(-decision * sigmoid(score), new FeatureVector(feats), grad)
          }
        }
      }
    }

    (ll, grad)
  }
}
