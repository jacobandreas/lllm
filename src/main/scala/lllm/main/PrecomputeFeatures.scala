package lllm.main

import breeze.stats.distributions.Uniform
import igor.experiment.Stage
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
class PrecomputeFeatures(val trainPath: String,
                         val order: Int = 3,
                         val featureGroupSize: Int = 1000,
                         val noiseSamples: Int = 10,
                         val useHashing: Boolean = false,
                         val objectiveType: ObjectiveType = CD) extends Stage {

  val includeWordIdentityInFeature = objectiveType != Hierarchical

  final val Featurizer = NGramFeaturizer(1, order, includeWordIdentityInFeature) +
                         WordAndIndexFeaturizer(includeWordIdentityInFeature) +
                         ConstantFeaturizer()

  override def run(): Unit = {

    val corpus = TextCorpusReader(trainPath)
    val featureCounts = task("counting features") { Counter(corpus.nGramIterator(order).flatMap { nGram => Featurizer(nGram).map((_,1)) }) }
    val posFeats = task("building feature index") { Index(corpus.nGramFeatureIndexer(order, Featurizer).filter(featureCounts(_) > 5)) }
    val feats = {
      if (useHashing)
        HashingFeatureIndex(posFeats)
      else
        posFeats
    }

    putDisk('FeatureIndex, feats)
    putDisk('Featurizer, Featurizer)
    putDisk('VocabIndex, corpus.vocabularyIndex)

    val counts = Counter[String,Double]
    corpus.nGramIterator(1).foreach { word => counts(word) += 1d }
    val noiseDistribution = Multinomial[Counter[String,Double],String](counts)
    //val normalizedCounts = counts / sum(counts)

    if (objectiveType == Hierarchical) {
      val huffmanDict = task("building Huffman dictionary") {
        counts.keysIterator.foreach { key => corpus.vocabularyIndex(key)}
        val intCounts = counts.keysIterator.map { key => (corpus.vocabularyIndex(key), counts(key))}.toIterable
        HuffmanDict.fromCounts(intCounts)
      }
      putDisk('HuffmanDict, huffmanDict)
    }

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

          assert(lines.isTraversableAgain)
          val wordIds: Seq[Int] = lines.flatMap { line =>
            line.split(" ") map corpus.vocabularyIndex
          }

          val (noiseFeatures, noiseProbs): (Seq[IndexedSeq[Array[Int]]], Seq[IndexedSeq[Double]]) = task("noise") { lines.flatMap { line =>
            line.split(" ").toIndexedSeq.nGrams(order).map { ngram =>
              val samples = if (objectiveType == NCE)
                              // TODO(jda) this should be a uniform sample
                              noiseDistribution.sample(noiseSamples)
                            else
                              noiseDistribution.sample(noiseSamples)
              samples.map { sample =>
                val noisedNGram = ngram.take(ngram.length - 1) :+ sample
                (Featurizer(noisedNGram).flatMap(feats.indexOpt).toArray, unigramModel.prob(ngram))
              }.unzip
            }
          }}.unzip

          putDisk(Symbol(s"DataFeatures$group"), dataFeatures)
          putDisk(Symbol(s"NoiseFeatures$group"), noiseFeatures)
          putDisk(Symbol(s"DataProbs$group"), dataProbs)
          putDisk(Symbol(s"NoiseProbs$group"), noiseProbs)
          putDisk(Symbol(s"WordIds$group"), wordIds)
        }
      }
    }

    putDisk('NLineGroups, Int.box(corpus.lineGroupIterator(featureGroupSize).length))
  }
}
