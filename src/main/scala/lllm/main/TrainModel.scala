package lllm.main

import breeze.features.FeatureVector
import breeze.linalg._
import breeze.numerics.{exp, log, log1p, sigmoid}
import breeze.optimize.BatchDiffFunction
import breeze.optimize.FirstOrderMinimizer.OptParams
import breeze.util.Index
import igor.experiment.{ResultCache, Stage}
import lllm.model.ObjectiveType._
import lllm.model.{LogLinearLanguageModel, UnigramLanguageModel}
import lllm.util.HuffmanDict

/**
 * @author jda
 */
object TrainModel extends Stage[LLLMParams] {

  override def run(config: LLLMParams, cache: ResultCache): Unit = {

    implicit val _c = config
    implicit val _r = cache

    val featureIndex:CrossProductIndex = cache.getDisk('CrossIndex)

    val optimization = OptParams(useStochastic = true, batchSize = 5, maxIterations = 500)

    val initTheta = config.objectiveType match {
      case Hierarchical =>
        val huffmanDict: HuffmanDict[Int] = cache.get('HuffmanDict)
        DenseVector.zeros[Double](featureIndex.size * huffmanDict.prefixIndex.size)
      case LowRank => DenseVector.rand[Double](featureIndex.size * config.rank)
      case _ => DenseVector.zeros[Double](featureIndex.size)
    }

    // TODO(jda) low rank is actually orthogonal to everything else---generalize later
    val objective = config.objectiveType match {
      case Hierarchical => makeObjectiveHierarchical
      case CD => makeObjectiveMonteCarlo
      case NCE => makeObjectiveNoiseContrastive
      case LowRank => makeObjectiveLowRank
      case _ => assert(false); makeObjectiveMonteCarlo
    }

    //GradientTester.test(objective, initTheta, toString = (x: Int) => x.toString, randFraction = 1)
    val optTheta = optimization.minimize(objective, initTheta)
    cache.put('Model, new LogLinearLanguageModel(cache.get('PredictionFeaturizer),
      cache.get('ContextFeaturizer),
      cache.getDisk('CrossIndex),
      optTheta))

  }

  // TODO(jda) refactor inner loop out of all of the following

  def makeObjectiveLowRank(implicit config: LLLMParams, cache: ResultCache): BatchDiffFunction[DenseVector[Double]] = {

    val featureIndex: Index[String] = cache.getDisk('CrossIndex)
    val vocabIndex: Index[String] = cache.getDisk('VocabIndex)

    new BatchDiffFunction[DenseVector[Double]] {

      override def fullRange: IndexedSeq[Int] = 0 until cache.get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {

        val matTheta = theta.asDenseMatrix.reshape(featureIndex.size, config.rank)
        // would be nice to precompute this, but we can't fit it in memory
        // val dots = matTheta * matTheta.t

        task("calculating") {

          val grad = DenseMatrix.zeros[Double](featureIndex.size, config.rank)
          var ll = 0d

          logger.info(s"batch: $batch")

          batch.foreach { batchIndex =>
            val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batchIndex"))
            val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = cache.getDisk(Symbol(s"NoiseFeatures$batchIndex"))

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

  def makeObjectiveHierarchical(implicit config: LLLMParams, cache: ResultCache): BatchDiffFunction[DenseVector[Double]] = {

    val featureIndex: Index[String] = cache.getDisk('CrossIndex)
    val huffmanDict: HuffmanDict[Int] = cache.get('HuffmanDict)

    new BatchDiffFunction[DenseVector[Double]] {

      override def fullRange: IndexedSeq[Int] = 0 until cache.get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {

        val matTheta = theta.asDenseMatrix.reshape(featureIndex.size, huffmanDict.prefixIndex.size)

        task("calculating") {

          val grad = DenseMatrix.zeros[Double](featureIndex.size, huffmanDict.prefixIndex.size)
          var ll = 0d

          batch.foreach { batchIndex =>
            val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batchIndex"))
            val batchWordIds: Seq[Int] = cache.getDisk(Symbol(s"WordIds$batchIndex"))

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

  def makeObjectiveMonteCarlo(implicit config: LLLMParams, cache: ResultCache): BatchDiffFunction[DenseVector[Double]] = {

    val featureIndex: Index[String] = cache.getDisk('CrossIndex)
    val vocabIndex: Index[String] = cache.getDisk('VocabIndex)

    new BatchDiffFunction[DenseVector[Double]] {

      override def fullRange: IndexedSeq[Int] = 0 until cache.get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {

        task("calculating") {

          val grad = DenseVector.zeros[Double](featureIndex.size)
          var ll = 0d

          logger.info(s"batch: $batch")

          batch.foreach { batchIndex =>
            val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batchIndex"))
            val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = cache.getDisk(Symbol(s"NoiseFeatures$batchIndex"))

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

  def makeObjectiveNoiseContrastive(implicit config: LLLMParams, cache: ResultCache): BatchDiffFunction[DenseVector[Double]] = {

    val featureIndex: Index[String] = cache.getDisk('CrossIndex)
    val vocabIndex: Index[String] = cache.getDisk('VocabIndex)
    val unigramModel: UnigramLanguageModel = cache.getDisk('UnigramModel)

    new BatchDiffFunction[DenseVector[Double]] {

      override def fullRange: IndexedSeq[Int] = 0 until cache.get('NLineGroups)

      override def calculate(theta: DenseVector[Double], batch: IndexedSeq[Int]): (Double, DenseVector[Double]) = {

        task("calculating") {

          val grad = DenseVector.zeros[Double](featureIndex.size)
          var ll = 0d

          logger.info(s"batch: $batch")

          batch.foreach { batchIndex =>
            val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batchIndex"))
            val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = cache.getDisk(Symbol(s"NoiseFeatures$batchIndex"))
            val batchPredictionProbs: Seq[Double] = cache.getDisk(Symbol(s"PredictionProbs$batchIndex"))
            val batchNoiseProbs: Seq[IndexedSeq[Double]] = cache.getDisk(Symbol(s"NoiseProbs$batchIndex"))

            (batchDataSamples zip batchPredictionProbs) zip (batchNoiseSamples zip batchNoiseProbs) foreach {
              case ((data, dataProb), (noise, noiseProbs)) =>

                val pData = sigmoid((theta dot new FeatureVector(data)) - log(config.noiseSamples * dataProb))
                val pNoise = noise zip noiseProbs map { case (n, np) => sigmoid((theta dot new FeatureVector(n)) - log(config.noiseSamples * np)) }

                ll += log(pData)
                ll += sum(pNoise.map(x => log1p(-x)))

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
