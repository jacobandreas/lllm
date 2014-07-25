package lllm.main

import breeze.stats.distributions.Uniform
import igor.experiment.{Experiment, Stage}
import erector.corpus.TextCorpusReader
import lllm.features.{HashingFeatureIndex, ConstantFeaturizer, NGramFeaturizer, WordAndIndexFeaturizer}
import erector.util.text.toNGramIterable
import breeze.linalg.{sum, Counter}
import lllm.model.UnigramLanguageModel
import breeze.util.Index
import breeze.features.FeatureVector
import lllm.util.HuffmanDict
import lllm.model.ObjectiveType._

/**
 * @author jda
 */
object PrecomputeFeatures extends Stage[LLLMParams] {

  override def run(config: LLLMParams, experiment: Experiment[LLLMParams]): Unit = {

    val includeWordIdentityInFeature = config.objectiveType != Hierarchical

    val Featurizer = NGramFeaturizer(1, config.order, includeWordIdentityInFeature) +
      WordAndIndexFeaturizer(includeWordIdentityInFeature) +
      ConstantFeaturizer(includeWordIdentityInFeature)

    val corpus = TextCorpusReader(config.trainPath)
    val featureCounts = task("counting features") { Counter(corpus.nGramIterator(config.order).flatMap { nGram => Featurizer(nGram).map((_,1)) }) }
    val posFeats = task("building feature index") { Index(corpus.nGramFeatureIndexer(config.order, Featurizer).filter(featureCounts(_) > 5)) }
    val feats = {
      if (config.useHashing)
        HashingFeatureIndex(posFeats)
      else
        posFeats
    }

    experiment.putDisk('FeatureIndex, feats)
    experiment.putDisk('Featurizer, Featurizer)
    experiment.putDisk('VocabIndex, corpus.vocabularyIndex)

    val counts = Counter[String,Double]
    corpus.nGramIterator(1).foreach { word => counts(word) += 1d }
    val noiseDistribution = Multinomial[Counter[String,Double],String](counts)
    //val normalizedCounts = counts / sum(counts)

    if (config.objectiveType == Hierarchical) {
      val huffmanDict = task("building Huffman dictionary") {
        counts.keysIterator.foreach { key => corpus.vocabularyIndex(key)}
        val intCounts = counts.keysIterator.map { key => (corpus.vocabularyIndex(key), counts(key))}.toIterable
        HuffmanDict.fromCounts(intCounts)
      }
      experiment.putDisk('HuffmanDict, huffmanDict)
    }

    val unigramModel = new UnigramLanguageModel(counts)
    experiment.putDisk('UnigramModel, unigramModel)

    task("caching features") {
      val lineGroups = corpus.lineGroupIterator(config.featureGroupSize)
      lineGroups.zipWithIndex.foreach { case (lines, group) =>

        task(s"batch $group") {
          val (dataFeatures, dataProbs): (Seq[Array[Int]], Seq[Double]) = task("data") { lines.flatMap { line =>
            line.split(" ").toIndexedSeq.nGrams(config.order).map { ngram =>
              (Featurizer(ngram).flatMap(feats.indexOpt).toArray, unigramModel.prob(ngram))
            }
          }}.unzip

          assert(lines.isTraversableAgain)
          val wordIds: Seq[Int] = lines.flatMap { line =>
            line.split(" ") map corpus.vocabularyIndex
          }

          val (noiseFeatures, noiseProbs): (Seq[IndexedSeq[Array[Int]]], Seq[IndexedSeq[Double]]) = task("noise") { lines.flatMap { line =>
            line.split(" ").toIndexedSeq.nGrams(config.order).map { ngram =>
              val samples = if (config.objectiveType == NCE)
                              // TODO(jda) this should be a uniform sample
                              noiseDistribution.sample(config.noiseSamples)
                            else
                              noiseDistribution.sample(config.noiseSamples)
              samples.map { sample =>
                val noisedNGram = ngram.take(ngram.length - 1) :+ sample
                (Featurizer(noisedNGram).flatMap(feats.indexOpt).toArray, unigramModel.prob(ngram))
              }.unzip
            }
          }}.unzip

          experiment.putDisk(Symbol(s"DataFeatures$group"), dataFeatures)
          experiment.putDisk(Symbol(s"NoiseFeatures$group"), noiseFeatures)
          experiment.putDisk(Symbol(s"DataProbs$group"), dataProbs)
          experiment.putDisk(Symbol(s"NoiseProbs$group"), noiseProbs)
          experiment.putDisk(Symbol(s"WordIds$group"), wordIds)
        }
      }
    }

    experiment.putDisk('NLineGroups, Int.box(corpus.lineGroupIterator(config.featureGroupSize).length))
  }
}
