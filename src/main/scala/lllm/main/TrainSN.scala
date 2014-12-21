package lllm.main

import java.io.{ObjectOutputStream, FileOutputStream}

import breeze.features.FeatureVector
import breeze.linalg.{SparseVector, sum, axpy, DenseVector}
import breeze.numerics.{exp, log}
import breeze.optimize.{GradientTester, BatchDiffFunction}
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.util.Index
import edu.berkeley.nlp.langmodel.{EnglishWordIndexer, NgramLanguageModel}
import erector.corpus.TextCorpusReader
import erector.learning.Feature
import igor.experiment.{ResultCache, Stage}
import lllm.features.Featurizer
import spire.syntax.cfor._
import erector.util.text.toNGramIterable
import collection.JavaConversions._
import scala.util.Random

/**
 * @author jda
 */
object TrainSN extends Stage[SelfNormalizingParams] {

  override def run(implicit config: SelfNormalizingParams, cache: ResultCache): Unit = {
    val productIndex: CrossProductIndex = cache.getDisk('productIndex)
    val initTheta = DenseVector.zeros[Double](productIndex.size)
    val numLineGroups: Int = cache.get('numLineGroups)

    val vocabIndex: Index[String] = cache.get('vocabularyIndex)
    val contextFeaturizer: Featurizer = cache.get('contextFeaturizer)
    val contextFeatureIndex: Index[Feature] = cache.get('contextFeatureIndex)

    val partitionFreq = (1d / config.partitionFrac).toInt

    val objective = new  BatchDiffFunction[DenseVector[Double]] {
      override def fullRange: IndexedSeq[Int] = 0 until numLineGroups
      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {
        val arrIndices = (0 until vocabIndex.size).map(Array(_)).toArray
        val batchIds = batch.toSet
        var ll = 0d
        val grad = DenseVector.zeros[Double](theta.length)
        val lines = TextCorpusReader(config.trainPath).prefix(config.numLines)
        lines.lineGroupIterator(config.lineGroupSize).zipWithIndex.filter(p => batchIds.contains(p._2)).foreach { case (batchLines, batchId) =>
          logger.info(s"batch $batchId")
//          val (bll, bgrad) = batchLines.par.map { line =>
          batchLines.foreach { line =>
            //          val (bll, bgrad) = batchLines.par.aggregate((0d, DenseVector.zeros[Double](theta.length)))({ (soFar, line) =>

            //            var lll = soFar._1
            //            val lgrad = soFar._2
            //            var lll = 0d
            //            val lgrad = SparseVector.zeros[Double](theta.length)
            // start symbol?
            val nGrams = (line.split(" ") :+ erector.util.text.DefaultStartSymbol).toIndexedSeq.nGrams(config.order).toArray
            cforRange(0 until nGrams.length) { iNGram =>
              val nGram = nGrams(iNGram)
              if (vocabIndex contains nGram.last) {
                val wordId = vocabIndex(nGram.last)
                val contextIds = contextFeaturizer(nGram).map(contextFeatureIndex)

                // score
                val topFeats = new FeatureVector(productIndex.crossProduct(arrIndices(wordId), contextIds))
                //                lll += theta dot topFeats
                //                axpy(1d, topFeats, lgrad)
                ll += theta dot topFeats
                axpy(1d, topFeats, grad)

                // normalizer
                //                val bottomFeats = Array.ofDim[FeatureVector](vocabIndex.size)
                //                val bottomScores = Array.ofDim[Double](vocabIndex.size)
                //                cforRange (0 until vocabIndex.size) { i =>
                //                  val fv = new FeatureVector(productIndex.crossProduct(Array(i), contextIds))
                //                  bottomFeats(i) = fv
                //                  bottomScores(i) = exp(theta dot fv)
                //                }
                //                val bottomScoresSum = sum(bottomScores)
                //                val invBottomScoresSum = 1 / sum(bottomScores)
                //                val logBottomScoresSum = log(bottomScoresSum)
                //                ll -= logBottomScoresSum
                //                cforRange (0 until bottomFeats.length) { i =>
                //                  axpy(-bottomScores(i) * invBottomScoresSum, bottomFeats(i), grad)
                //                }

                //                val partitionPenalty = bottomScoresSum - 1
                //                ll -= config.partitionL2 * 0.5 * partitionPenalty * partitionPenalty
                //                cforRange (0 until bottomFeats.length) { i =>
                //                  axpy(-(bottomScores(i) * invBottomScoresSum + config.partitionL2 * partitionPenalty * bottomScores(i)), bottomFeats(i), grad)
                //                }

                // partition penalty
                if (Random.nextDouble() < config.partitionFrac) {
                  //                if (true) {
                  val bottomFeats = Array.ofDim[FeatureVector](vocabIndex.size)
                  val bottomScores = Array.ofDim[Double](vocabIndex.size)
                  cforRange(0 until vocabIndex.size) { i =>
                    val ai = arrIndices(i)
                    val fv = new FeatureVector(productIndex.crossProduct(ai, contextIds))
                    bottomFeats(i) = fv
                    bottomScores(i) = exp(theta dot fv)
                  }
                  val bottomScoresSum = sum(bottomScores)
                  val bottomScoresSumInv = 1d / bottomScoresSum
                  val logBottomScoresSum = log(bottomScoresSum)
                  val partitionPenalty = logBottomScoresSum
                  //                  lll -= config.partitionL2 * partitionFreq * 0.5 * partitionPenalty * partitionPenalty
                  ll -= config.partitionL2 * partitionFreq * 0.5 * partitionPenalty * partitionPenalty
                  cforRange(0 until bottomFeats.length) { i =>
                    //                    axpy(-config.partitionL2 * partitionFreq * logBottomScoresSum * bottomScoresSumInv * bottomScores(i), bottomFeats(i), lgrad)
                    axpy(-config.partitionL2 * partitionFreq * logBottomScoresSum * bottomScoresSumInv * bottomScores(i), bottomFeats(i), grad)
                  }
                }
              }
            }
          }
//            (lll, lgrad)
//          }, { (p1, p2) => (p1._1 + p2._1, p1._2 + p2._2) })
//          ll += bll
//          grad += bgrad
        }
        (-ll, -grad)
      }
    }


    logger.info(s"I will do ${numLineGroups * config.numEpochs / 256} iterations")
    val opt = OptParams(
      useStochastic = true,
      batchSize = 256,
      maxIterations = numLineGroups * config.numEpochs / 256,
      regularization = config.paramL2
    )

    val optTheta = opt.minimize(objective, initTheta)
//    GradientTester.test(objective, optTheta, toString = (x:Int) => x.toString)

    val indexer = EnglishWordIndexer.getIndexer
    vocabIndex.foreach(indexer.add)
//    val fullVocabIndex = Index(indexer.iterator)

    val optModel = new NgramLanguageModel with Serializable {
      lazy val indexer = EnglishWordIndexer.getIndexer
      override val getOrder: Int = 3
      override def getCount(ngram: Array[Int]): Long = 0
      override def getNgramLogProbability(ngram: Array[Int], from: Int, to: Int): Double = {
        val wordId = ngram(to-1)
        if (wordId >= vocabIndex.size) {
          Double.NegativeInfinity
        } else {
//          println(ngram.slice(from,to).toSeq)
//          val words = ngram.slice(from,to).map(i => if (i == -1) "UNK" else indexer.get(i))
          val words = ngram.slice(from,to).filterNot(_ == -1).map(indexer.get)
          val contextIds = contextFeaturizer(words).map(contextFeatureIndex)
          val feats = new FeatureVector(productIndex.crossProduct(Array(wordId), contextIds))
//          println(feats)
          optTheta dot feats
        }
      }
    }

    new ObjectOutputStream(new FileOutputStream("optModel.ser")).writeObject(optModel)
  }

}
