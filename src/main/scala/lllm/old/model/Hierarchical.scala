package lllm.old.model

import breeze.features.FeatureVector
import breeze.linalg._
import breeze.numerics._
import breeze.util.Index
import erector.learning._
import igor.experiment.ResultCache
import lllm.features.Featurizer
import lllm.main.CrossProductIndex
import lllm.old.main.{LLLMParams, PrecomputeFeatures}
import lllm.util.HuffmanDict

/**
 * @author jda
 */
case object Hierarchical extends Objective {

  override def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int = {
    numFeatures //* cache.get[HuffmanDict[Int]]('HuffmanDict).prefixIndex.size
  }

  override def apply(theta: DenseVector[Double],
                     batch: Int,
                     batchLines: Seq[String],
                     featureIndex: Index[String])
                    (implicit config: LLLMParams,
                     cache: ResultCache): (Double, DenseVector[Double]) = {

    //val theta = vecTheta.asDenseMatrix.reshape(featureIndex.size, config.rank)
    var ll = 0d
    val grad = DenseVector.zeros[Double](theta.length)

    val huffmanDict: HuffmanDict[Int] = cache.get('HuffmanDict)
    val cpFeatureIndex = featureIndex.asInstanceOf[CrossProductIndex]

//    val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
//    val batchWordIds: Seq[Int] = cache.getDisk(Symbol(s"WordIds$batch"))

    assert(!config.cacheFeatures)
    val contextFeaturizer: Featurizer = cache.get('ContextFeaturizer)
    val featIndex: Index[Feature] = cache.get('FeatIndex)
    val vocabIndex: Index[String] = cache.get('VocabIndex)
    //val lines: Source = cache.readFile(Symbol(s"Lines$batch"))
    //val batchNGrams = PrecomputeFeatures.makeBatchNGrams(lines.reset().getLines().toIterable)
    val batchNGrams = PrecomputeFeatures.makeBatchNGrams(batchLines)
    val batchContextFeatures = PrecomputeFeatures.makeBatchContextFeatures(batchNGrams, contextFeaturizer, featIndex)

    (batchNGrams zip batchContextFeatures).foreach { case (ngram, contextFeatures) =>
      val wordId = vocabIndex(ngram.last)
//      println(wordId)
//      println(ngram.last)
      val code = huffmanDict.dict.get(wordId).get
      // force evaluation now
      code.tails.toArray.filter(_.nonEmpty).foreach { prefix =>
        val decision = if (prefix.head == '1') 1d else -1d
        val history = prefix.tail
        val nodeId = huffmanDict.prefixIndex(history)
        val feats = cpFeatureIndex.crossProduct(Array(nodeId), contextFeatures)
        val score = (theta dot new FeatureVector(feats)) * decision
        ll += -log1p(exp(score))
        axpy(-decision * sigmoid(score), new FeatureVector(feats), grad)
        //println(-log(1 + exp(score)))
      }
    }
    (ll, grad)

//    batchDataSamples zip batchWordIds foreach { case (feats, word) =>
//      val code = huffmanDict.dict.get(word).get
//      (0 until code.length) foreach { prefixLength =>
//
//        val decision = if(code.head) 1d else -1d
//        val history = code.tail
//
//        val nodeId = huffmanDict.prefixIndex(history)
//        //val thetaNode = theta(::, nodeId)
//
//        val denomFeats = cpFeatureIndex.crossProduct(new Array(nodeId), context)
//        denomScores(i) = exp(theta dot new FeatureVector(denomFeats))
//
//        val score = (thetaNode dot new FeatureVector(feats)) * decision
//
//        ll += log(1 + exp(score))
//        axpy(sigmoid(-score), new FeatureVector(feats), grad(::, nodeId))
//      }
//    }
//    (ll, grad.toDenseVector)
  }
}
