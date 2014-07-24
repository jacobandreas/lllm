package lllm.main

import igor.experiment.Stage
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.optimize.{DiffFunction, GradientTester, BatchDiffFunction}
import breeze.linalg._
import breeze.util.Index
import breeze.features.FeatureVector
import breeze.numerics.{sigmoid, log, exp}
import breeze.optimize.FirstOrderMinimizer.OptParams
import lllm.features.HashingFeatureIndex
import lllm.model.ObjectiveType._
import lllm.model.{UnigramLanguageModel, LogLinearLanguageModel}
import scala.collection.mutable.ArrayBuffer
import scala.collection.mutable
import lllm.util.HuffmanDict

/**
 * @author jda
 */
// TODO(jda): for erector: default params should be set centrally
class TrainModel(noiseSamples: Int = 10,
                 useHashing: Boolean = false,
                 objectiveType: ObjectiveType = LowRank,
                 rank: Int = 10) extends Stage {

  override def run(): Unit = {

    val featureIndex: Index[String] = getDisk('FeatureIndex)

    val optimization = OptParams(useStochastic = true, batchSize = 5, maxIterations = 500)

    val initTheta = objectiveType match {
      case Hierarchical =>
        val huffmanDict: HuffmanDict[Int] = get('HuffmanDict)
        DenseVector.zeros[Double](featureIndex.size * huffmanDict.prefixIndex.size)
      case LowRank => DenseVector.rand[Double](featureIndex.size * rank)
      case _ => DenseVector.zeros[Double](featureIndex.size)
    }

    // TODO(jda) low rank is actually orthogonal to everything else---generalize later
    val objective = objectiveType match {
      case Hierarchical => makeObjectiveHierarchical
      case CD => makeObjectiveMonteCarlo
      case NCE => makeObjectiveNoiseContrastive
      case LowRank => makeObjectiveLowRank
      case _ => { assert(false); makeObjectiveMonteCarlo }
    }

    //GradientTester.test(objective, initTheta, toString = (x: Int) => x.toString, randFraction = 1)
    val optTheta = optimization.minimize(objective, initTheta)
    put('Model, new LogLinearLanguageModel(get('Featurizer), get('FeatureIndex), get('VocabIndex), optTheta))

  }

  // TODO(jda) refactor inner loop out of all of the following

  def makeObjectiveLowRank: BatchDiffFunction[DenseVector[Double]] = {

    val featureIndex: Index[String] = getDisk('FeatureIndex)
    val vocabIndex: Index[String] = getDisk('VocabIndex)

    new BatchDiffFunction[DenseVector[Double]] {

      override def fullRange: IndexedSeq[Int] = 0 until get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {

        val matTheta = theta.asDenseMatrix.reshape(featureIndex.size, rank)
        // would be nice to precompute this, but we can't fit it in memory
        // val dots = matTheta * matTheta.t

        task("calculating") {

          val grad = DenseMatrix.zeros[Double](featureIndex.size, rank)
          var ll = 0d

          logger.info(s"batch: $batch")

          batch.foreach { batchIndex =>
            val batchDataSamples: Seq[Array[Int]] = getDisk(Symbol(s"DataFeatures$batchIndex"))
            val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = getDisk(Symbol(s"NoiseFeatures$batchIndex"))

            batchDataSamples zip batchNoiseSamples foreach { case (data, noise) =>

              // TODO(jda) every pair of for loops does twice as much work as necessary
              var score = 0d
              data.foreach { featI =>
                data.foreach { featJ =>
                  //score += dots(featI, featJ)
                  score += matTheta(featI,::).t dot matTheta(featJ,::).t
                }
              }
              val denomScores = (noise :+ data) map { sample: Array[Int] =>
                var sampleScore = 0d
                sample.foreach { featI =>
                  sample.foreach { featJ =>
                    //sampleScore += dots(featI, featJ)
                    sampleScore += matTheta(featI,::).t dot matTheta(featJ,::).t
                  }
                }
                exp(sampleScore)
              }

              val norm = sum(denomScores)

              ll += score
              ll -= log(norm)

              data.foreach { featI =>
                data.foreach { featJ =>
                  axpy(2d, matTheta(featJ,::).t, grad(featI,::).t)
                }
              }
              denomScores zip (noise :+ data) foreach { case (dScore, dFeat) =>
                dFeat.foreach { featI =>
                  dFeat.foreach { featJ =>
                    axpy(-2d * dScore / norm, matTheta(featJ,::).t, grad(featI,::).t)
                  }
                }
              }
            }
          }
          //logger.info(grad.toString)
          (-ll, -grad.toDenseVector)
        }
      }
    }
  }

  def makeObjectiveHierarchical: BatchDiffFunction[DenseVector[Double]] = {

    val featureIndex: Index[String] = getDisk('FeatureIndex)
    val huffmanDict: HuffmanDict[Int] = get('HuffmanDict)

    new BatchDiffFunction[DenseVector[Double]] {

      override def fullRange: IndexedSeq[Int] = 0 until get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {

        val matTheta = theta.asDenseMatrix.reshape(featureIndex.size, huffmanDict.prefixIndex.size)

        task("calculating") {

          val grad = DenseMatrix.zeros[Double](featureIndex.size, huffmanDict.prefixIndex.size)
          var ll = 0d

          batch.foreach { batchIndex =>
            val batchDataSamples: Seq[Array[Int]] = getDisk(Symbol(s"DataFeatures$batchIndex"))
            val batchWordIds: Seq[Int] = getDisk(Symbol(s"WordIds$batchIndex"))

            batchDataSamples zip batchWordIds foreach { case (feats, word) =>
              val code = huffmanDict.dict.get(word).get
              (0 until code.length) foreach { prefixLength =>

                val decision = if(code.head) 1d else -1d
                val history = code.tail

                val nodeId = huffmanDict.prefixIndex(history)
                val thetaNode = matTheta(::, nodeId)

                val score = (thetaNode dot new FeatureVector(feats)) * decision

                ll += log(1 + exp(score))
                axpy(sigmoid(-score), new FeatureVector(feats), thetaNode)
              }
            }
          }
          (-ll, -grad.toDenseVector)
        }
      }
    }
  }

  def makeObjectiveMonteCarlo: BatchDiffFunction[DenseVector[Double]] = {

    val featureIndex: Index[String] = getDisk('FeatureIndex)
    val vocabIndex: Index[String] = getDisk('VocabIndex)

    new BatchDiffFunction[DenseVector[Double]] {

      override def fullRange: IndexedSeq[Int] = 0 until get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {

        task("calculating") {

          val grad = DenseVector.zeros[Double](featureIndex.size)
          var ll = 0d

          logger.info(s"batch: $batch")

          batch.foreach { batchIndex =>
            val batchDataSamples: Seq[Array[Int]] = getDisk(Symbol(s"DataFeatures$batchIndex"))
            val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = getDisk(Symbol(s"NoiseFeatures$batchIndex"))

            batchDataSamples zip batchNoiseSamples foreach { case (data, noise) =>

              // here we're just using the "noise" samples from NCE to get a monte carlo estimate of the partition
              // this is wrong because the noise samples are non-uniform---just a quick hack until we add NCE

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
              noiseExpScores zip (noise :+ data) foreach { case (dScore, dFeat) => axpy(-dScore / norm, new FeatureVector(dFeat), grad) }
            }
          }
          (-ll, -grad)
        }
      }
    }
  }

  def makeObjectiveNoiseContrastive: BatchDiffFunction[DenseVector[Double]] = {

    val featureIndex: Index[String] = getDisk('FeatureIndex)
    val vocabIndex: Index[String] = getDisk('VocabIndex)
    val unigramModel: UnigramLanguageModel = getDisk('UnigramModel)

    new BatchDiffFunction[DenseVector[Double]] {

      override def fullRange: IndexedSeq[Int] = 0 until get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {

        task("calculating") {

          val grad = DenseVector.zeros[Double](featureIndex.size)
          var ll = 0d

          logger.info(s"batch: $batch")

          batch.foreach { batchIndex =>
            val batchDataSamples: Seq[Array[Int]] = getDisk(Symbol(s"DataFeatures$batchIndex"))
            val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = getDisk(Symbol(s"NoiseFeatures$batchIndex"))
            val batchDataProbs: Seq[Double] = getDisk(Symbol(s"DataProbs$batchIndex"))
            val batchNoiseProbs: Seq[IndexedSeq[Double]] = getDisk(Symbol(s"NoiseProbs$batchIndex"))

            (batchDataSamples zip batchDataProbs) zip (batchNoiseSamples zip batchNoiseProbs) foreach {
              case ((data, dataProb), (noise, noiseProbs)) =>

                val pData = sigmoid((theta dot new FeatureVector(data)) - log(noiseSamples * dataProb))
                val pNoise = noise zip noiseProbs map { case (n, np) => sigmoid((theta dot new FeatureVector(n)) - log(noiseSamples * np)) }

                ll += log(pData)
                ll += sum(pNoise.map(x => log(1 - x)))

                axpy(1 - pData, new FeatureVector(data), grad)
                pNoise zip noise foreach { case (p, n) => axpy(-p, new FeatureVector(n), grad) }
            }
          }
          (-ll, -grad)
        }
      }
    }
  }
}
