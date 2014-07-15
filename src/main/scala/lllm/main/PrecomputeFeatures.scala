package lllm.main

import igor.experiment.Stage
import erector.corpus.TextCorpusReader
import lllm.features.{ConstantFeaturizer, NGramFeaturizer, WordAndIndexFeaturizer}
import erector.util.text.toNGramIterable
import breeze.linalg.{sum, Counter}
import lllm.model.UnigramLanguageModel
import breeze.util.Index
import breeze.features.FeatureVector

/**
 * @author jda
 */
class PrecomputeFeatures(val trainPath: String,
                         val order: Int = 3,
                         val featureGroupSize: Int = 1000,
                         val noiseSamples: Int = 10) extends Stage {

  //final val LineGroupSize = 3000
  final val Featurizer = NGramFeaturizer(1, order) + WordAndIndexFeaturizer() + ConstantFeaturizer()
  //final val NoiseSamples = 10

  override def run(): Unit = {

    val corpus = TextCorpusReader(trainPath)
    val featureCounts = task("counting features") { Counter(corpus.nGramIterator(order).flatMap { nGram => Featurizer(nGram).map((_,1)) }) }
    //logger.info(featureCounts.toString)
    val feats = task("building feature index") { Index(corpus.nGramFeatureIndexer(order, Featurizer).filter(featureCounts(_) > 5)) }
    //logger.info(feats.toString)

    putDisk('FeatureIndex, feats)
    putDisk('Featurizer, Featurizer)
    putDisk('VocabIndex, corpus.vocabularyIndex)

    val counts = Counter[String,Double]
    corpus.nGramIterator(1).foreach { word => counts(word) += 1d }
    val noiseDistribution = Multinomial[Counter[String,Double],String](counts)
    //val normalizedCounts = counts / sum(counts)

    val unigramModel = new UnigramLanguageModel(counts)
    putDisk('UnigramModel, unigramModel)

    task("caching features") {
      val lineGroups = corpus.lineGroupIterator(featureGroupSize)
      lineGroups.zipWithIndex.foreach { case (lines, group) =>
        task(s"batch $group") {
          val (dataFeatures, dataProbs): (Seq[Array[Int]], Seq[Double]) = task("data") { lines.flatMap { line =>
            line.split(" ").toIndexedSeq.nGrams(order).map { ngram =>
              (Featurizer(ngram).flatMap(feats.indexOpt).toArray, unigramModel.prob(ngram))
            }
          }}.unzip

          val (noiseFeatures, noiseProbs): (Seq[IndexedSeq[Array[Int]]], Seq[IndexedSeq[Double]]) = task("noise") { lines.flatMap { line =>
            line.split(" ").toIndexedSeq.nGrams(order).map { ngram =>
              val samples = noiseDistribution.sample(noiseSamples) :+ ngram(ngram.length - 1)
              samples.map { sample =>
                val noisedNGram = ngram.take(ngram.length - 1) :+ sample
                //logger.info(noisedNGram.toString)
                // TODO(jda) use negative feature hashing instead of just throwing these away
                (Featurizer(noisedNGram).flatMap(feats.indexOpt).toArray, unigramModel.prob(ngram))
              }.unzip
            }
          }}.unzip

          putDisk(Symbol(s"DataFeatures$group"), dataFeatures)
          putDisk(Symbol(s"NoiseFeatures$group"), noiseFeatures)
          putDisk(Symbol(s"DataProbs$group"), dataProbs)
          putDisk(Symbol(s"NoiseProbs$group"), noiseProbs)
        }
      }
    }

    putDisk('NLineGroups, Int.box(corpus.lineGroupIterator(featureGroupSize).length))
  }
}
