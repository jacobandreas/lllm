package lllm.old.model

import breeze.features.FeatureVector
import breeze.linalg.{Counter, DenseVector}
import breeze.numerics._
import breeze.linalg.{sum,axpy}
import breeze.stats.distributions.Multinomial
import breeze.util.Index
import igor.experiment.ResultCache
import lllm.old.main.{LLLMParams, PrecomputeFeatures}

import scala.io.Source

/**
 * @author jda
 */
case object NCE extends Objective {

  override def numParams(numFeatures: Int)(implicit config: LLLMParams, cache: ResultCache): Int = numFeatures

  override def apply(theta: DenseVector[Double],
                     batch: Int,
                     batchLines: Seq[String],
                     featureIndex: Index[String])
                    (implicit config: LLLMParams,
                     cache: ResultCache): (Double, DenseVector[Double]) = {

    var ll = 0d
    val grad = DenseVector.zeros[Double](theta.length)

//    val batchDataSamples: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
//    val batchNoiseSamples: Seq[IndexedSeq[Array[Int]]] = cache.getDisk(Symbol(s"NoiseFeatures$batch"))
//    val batchPredictionProbs: Seq[Double] = cache.getDisk(Symbol(s"PredictionProbs$batch"))
//    val batchNoiseProbs: Seq[IndexedSeq[Double]] = cache.getDisk(Symbol(s"NoiseProbs$batch"))

    val (batchDataSamples, batchNoiseSamples, batchPredictionProbs, batchNoiseProbs) =
      if (config.cacheFeatures) {
        val ds: Seq[Array[Int]] = cache.getDisk(Symbol(s"PredictionFeatures$batch"))
        val ns: Seq[IndexedSeq[Array[Int]]] = cache.getDisk(Symbol(s"NoiseFeatures$batch"))
        val pp: Seq[Double] = cache.getDisk(Symbol(s"PredictionProbs$batch"))
        val np: Seq[IndexedSeq[Double]] = cache.getDisk(Symbol(s"NoiseProbs$batch"))
        (ds, ns, pp, np)
      } else {
        val samplingDistribution = Multinomial(cache.get[Counter[String,Double]]('SamplingDistributionParams))
        //val lines: Seq[String] = cache.getDisk(Symbol(s"Lines$batch"))
        val lines: Source = cache.readFile(Symbol(s"Lines$batch"))
        val (ds, pp) = PrecomputeFeatures.recomputeBatchDataFeatures(lines.getLines().toIterable, cache.get('ContextFeaturizer), cache.get('PredictionFeaturizer), cache.get('PredictionPreprocessor), cache.get('FeatIndex), cache.get('VocabIndex), cache.get('CrossIndex), samplingDistribution)
        val (ns, np) = PrecomputeFeatures.recomputeBatchNoiseFeatures(lines.reset().getLines().toIterable, cache.get('ContextFeaturizer), cache.get('PredictionFeaturizer), cache.get('PredictionPreprocessor), cache.get('FeatIndex), cache.get('VocabIndex), cache.get('CrossIndex), samplingDistribution)
        (ds, ns, pp, np)
      }

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
