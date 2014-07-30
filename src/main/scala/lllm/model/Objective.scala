package lllm.model

import breeze.features.FeatureVector
import breeze.linalg.{sum, DenseMatrix, DenseVector, axpy}
import breeze.numerics.{log1p, sigmoid, log, exp}
import breeze.util.Index
import igor.experiment.ResultCache
import lllm.features.Featurizer
import lllm.main.{CrossProductIndex, LLLMParams}
import lllm.util.HuffmanDict

/**
 * @author jda
 */

// TODO(jda) each objective should get its own file

sealed trait Objective {

  def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int

  def apply(theta: DenseVector[Double],
            batch: Int,
            featureIndex: Index[String])
           (implicit config: LLLMParams,
            cache: ResultCache): (Double, DenseVector[Double])

}

case object Exact extends Objective {

  override def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int = numFeatures

  override def apply(theta: DenseVector[Double],
                     batch: Int,
                     featureIndex: Index[String])
                    (implicit config: LLLMParams,
                     cache: ResultCache): (Double, DenseVector[Double]) = {
    var ll = 0d
    val grad = DenseVector.zeros[Double](theta.length)

    // TODO(jda) everyone knows it's a CrossProductIndex
    val cpFeatureIndex = featureIndex.asInstanceOf[CrossProductIndex]
    val vocabIndex: Index[String] = cache.get('VocabIndex)
    val batchContextSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"ContextFeatures$batch"))
    val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
    val predictionFeaturizer: Featurizer = cache.getDisk('PredictionFeaturizer)

    val vocabPredFeats: Array[Array[Int]] = vocabIndex.map(w => predictionFeaturizer(IndexedSeq(w)).map(vocabIndex).toArray).toArray

    (batchContextSamples zip batchDataSamples) foreach { case (context, data) =>
      val fvData = new FeatureVector(data)
      val score = theta dot fvData
      ll += score
      axpy(1d, fvData, grad)

      val denomScores = new Array[Double](vocabPredFeats.size)
      var i = 0
      while (i < denomScores.length) {
        val denomFeats = cpFeatureIndex.crossProduct(vocabPredFeats(i), context)
        denomScores(i) = exp(theta dot new FeatureVector(denomFeats))
        i += 1
      }
      val norm = sum(denomScores)
      ll -= log(norm)
      i = 0
      while (i < denomScores.length) {
        // TODO(jda) better to pre-cache this?
        val denomFeats = cpFeatureIndex.crossProduct(vocabPredFeats(i), context)
        axpy(-denomScores(i) / norm, new FeatureVector(denomFeats), grad)
        i += 1
      }
    }
    (ll, grad)
  }

}

case object CD extends Objective {

  override def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int = numFeatures

  override def apply(theta: DenseVector[Double],
            batch: Int,
            featureIndex: Index[String])
           (implicit config: LLLMParams,
            cache: ResultCache): (Double, DenseVector[Double]) = {

    var ll = 0d
    val grad = DenseVector.zeros[Double](theta.length)

    val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
    val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = cache.getDisk(Symbol(s"NoiseFeatures$batch"))

    batchDataSamples zip batchNoiseSamples foreach { case (data, noise) =>

      // here we're just using the "noise" samples from to get a monte carlo estimate of the partition

      // there doesn't seem to be a significant performance difference between the following two impls:

      // "fast"

      //              val fvData = new FeatureVector(data)
      //              val score = theta dot fvData
      //              ll += score
      //              axpy(1d, fvData, grad)
      //
      //              //val noiseExpScores = new ArrayBuffer[Double](noise.size + 1)
      //              val noiseExpScores = new Array[Double](noise.size + 1)
      //              var i = 0
      //              while (i < noise.size) {
      //                noiseExpScores.update(i, exp(theta dot new FeatureVector(noise(i))))
      //                i += 1
      //              }
      //              noiseExpScores.update(noise.size, exp(score))
      //              val norm = sum(noiseExpScores) / noiseExpScores.length * vocabIndex.size
      //              ll -= log(norm)
      //
      //              i = 0
      //              while (i < noise.size) {
      //                axpy(-noiseExpScores(i) / norm, new FeatureVector(noise(i)), grad)
      //                i += 1
      //              }
      //              axpy(-noiseExpScores(noise.size) / norm, fvData, grad)


      // "slow"

      val score = theta dot new FeatureVector(data)
      val noiseExpScores = (noise :+ data).map(x => exp(theta dot new FeatureVector(x))).toSeq
      val norm = sum(noiseExpScores) // just a constant factor to make denom the right size // / (1 + noise.length) * vocabIndex.size

      ll += score
      ll -= log(norm)
      //assert(score - log(norm) <= 0)

      axpy(1d, new FeatureVector(data), grad)
      noiseExpScores zip (noise :+ data) foreach { case (dScore, dFeat) => axpy(-dScore / norm, new FeatureVector(dFeat), grad)}
    }
    (ll, grad)
  }

}

case object NCE extends Objective {

  override def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int = numFeatures

  override def apply(theta: DenseVector[Double],
                     batch: Int,
                     featureIndex: Index[String])
                    (implicit config: LLLMParams,
                     cache: ResultCache): (Double, DenseVector[Double]) = {

    var ll = 0d
    val grad = DenseVector.zeros[Double](theta.length)

    val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
    val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = cache.getDisk(Symbol(s"NoiseFeatures$batch"))
    val batchPredictionProbs: Seq[Double] = cache.getDisk(Symbol(s"PredictionProbs$batch"))
    val batchNoiseProbs: Seq[IndexedSeq[Double]] = cache.getDisk(Symbol(s"NoiseProbs$batch"))

    (batchDataSamples zip batchPredictionProbs) zip (batchNoiseSamples zip batchNoiseProbs) foreach {
      case ((data, dataProb), (noise, noiseProbs)) =>

        val pData = sigmoid((theta dot new FeatureVector(data)) - log(config.noiseSamples) - dataProb)
        val pNoise = noise zip noiseProbs map { case (n, np) => sigmoid((theta dot new FeatureVector(n)) - log(config.noiseSamples) - np)}
        //println(noiseProbs)

        ll += log(pData)
        ll += sum(pNoise.map(x => log1p(-x)))

//        println(pData)
//        println(pNoise)
//        println(log(pData))
//        println(sum(pNoise.map(x => log1p(-x))))
//        println()

        axpy(1 - pData, new FeatureVector(data), grad)
        pNoise zip noise foreach { case (p, n) => axpy(-p, new FeatureVector(n), grad)}
    }
    (ll, grad)
  }
}

// TODO this does not play nicely with the cross-product repr
case object Hierarchical extends Objective {

  override def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int = {
    numFeatures * cache.get[HuffmanDict[Int]]('HuffmanDict).prefixIndex.size
  }

  override def apply(vecTheta: DenseVector[Double],
                     batch: Int,
                     featureIndex: Index[String])
                    (implicit config: LLLMParams,
                     cache: ResultCache): (Double, DenseVector[Double]) = {

    val theta = vecTheta.asDenseMatrix.reshape(featureIndex.size, config.rank)
    var ll = 0d
    val grad = DenseMatrix.zeros[Double](featureIndex.size, config.rank)

    val huffmanDict: HuffmanDict[Int] = cache.get('HuffmanDict)

    val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
    val batchWordIds: Seq[Int] = cache.getDisk(Symbol(s"WordIds$batch"))

    batchDataSamples zip batchWordIds foreach { case (feats, word) =>
      val code = huffmanDict.dict.get(word).get
      (0 until code.length) foreach { prefixLength =>

        val decision = if(code.head) 1d else -1d
        val history = code.tail

        val nodeId = huffmanDict.prefixIndex(history)
        val thetaNode = theta(::, nodeId)

        val score = (thetaNode dot new FeatureVector(feats)) * decision

        ll += log(1 + exp(score))
        axpy(sigmoid(-score), new FeatureVector(feats), grad(::, nodeId))
      }
    }
    (ll, grad.toDenseVector)
  }
}

case object LowRank extends Objective {

  override def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int = {
    numFeatures * config.rank
  }

  override def apply(vecTheta: DenseVector[Double],
                     batch: Int,
                     featureIndex: Index[String])
                    (implicit config: LLLMParams,
                     cache: ResultCache): (Double, DenseVector[Double]) = {

    val theta = vecTheta.asDenseMatrix.reshape(featureIndex.size, config.rank)
    var ll = 0d
    val grad = DenseMatrix.zeros[Double](featureIndex.size, config.rank)

    val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
    val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = cache.getDisk(Symbol(s"NoiseFeatures$batch"))

    batchDataSamples zip batchNoiseSamples foreach { case (data, noise) =>

      // TODO(jda) every pair of for loops does twice as much work as necessary
      var score = 0d
      data.foreach { featI =>
        data.foreach { featJ =>
          //score += dots(featI, featJ)
          score += theta(featI,::).t dot theta(featJ,::).t
        }
      }
      val denomScores = (noise :+ data) map { sample: Array[Int] =>
        var sampleScore = 0d
        sample.foreach { featI =>
          sample.foreach { featJ =>
            //sampleScore += dots(featI, featJ)
            sampleScore += theta(featI,::).t dot theta(featJ,::).t
          }
        }
        exp(sampleScore)
      }

      val norm = sum(denomScores)

      ll += score
      ll -= log(norm)

      data.foreach { featI =>
        data.foreach { featJ =>
          axpy(2d, theta(featJ,::).t, grad(featI,::).t)
        }
      }
      denomScores zip (noise :+ data) foreach { case (dScore, dFeat) =>
        dFeat.foreach { featI =>
          dFeat.foreach { featJ =>
            axpy(-2d * dScore / norm, theta(featJ,::).t, grad(featI,::).t)
          }
        }
      }
    }
    (ll, grad.toDenseVector)
  }
}