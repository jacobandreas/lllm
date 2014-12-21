package lllm.model

import breeze.features.FeatureVector
import breeze.linalg.DenseVector
import breeze.numerics.{exp, log1p, Inf}
import breeze.util.Index
import lllm.features.Featurizer
import lllm.main.CrossProductIndex
import lllm.util.HuffmanDict

/**
 * @author jda
 */

class HierarchicalLanguageModel(contextFeaturizer: Featurizer,
                                val featureIndex: CrossProductIndex,
                                vocabIndex: Index[String],
                                huffmanDict: HuffmanDict[Int],
                                val theta: DenseVector[Double]) extends LanguageModel with Serializable {
  def logProb(ngram: IndexedSeq[String]): Double = {
    val contextIds = contextFeaturizer(ngram).flatMap(featureIndex.secondIndex.indexOpt).toArray
    if (!vocabIndex.contains(ngram.last)) return -Inf
    val wordId: Int = vocabIndex(ngram.last)
    //    print(ngram.last)
    //    print(wordId)
    //    print(huffmanDict.dict)
    if (!huffmanDict.dict.contains(wordId)) return -Inf
    val code = huffmanDict.dict.get(wordId).get
    var score = 0d
    // TODO(jda) de-HOF-ify
    code.tails.toArray.filter(_.nonEmpty).foreach { prefix =>
      val decision = if (prefix.head == '1') 1d else -1d
      val history = prefix.tail
      val decisionIds = Array(huffmanDict.prefixIndex(history), huffmanDict.constIndex)
      val feats = featureIndex.crossProduct(decisionIds, contextIds)
      score += -log1p(exp((theta dot new FeatureVector(feats)) * decision))
    }
    //println("giving back", score)
    score
  }

  def prob(ngram: IndexedSeq[String]): Double = exp(logProb(ngram))
}
