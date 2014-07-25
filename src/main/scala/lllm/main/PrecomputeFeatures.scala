package lllm.main

import breeze.stats.distributions.Uniform
import igor.experiment.{Experiment, Stage}
import erector.corpus.TextCorpusReader
import lllm.features._
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

    val contextFeaturizer = NGramFeaturizer(2, config.order) + WordAndIndexFeaturizer() + ConstantFeaturizer()
    val predictionFeaturizer = IdentityFeaturizer()
    // we assume for now that the features that come out of predictionFeaturizer look _exactly like_ the keys in the
    // vocabulary index below

    val productFeaturizer = contextFeaturizer * predictionFeaturizer

    val corpus = TextCorpusReader(config.trainPath)
    val contextFeatureCounts = Counter(corpus.nGramIterator(config.order).flatMap { nGram => contextFeaturizer(nGram).map((_,1)) })
    val positiveFeatIndex = Index(corpus.nGramFeatureIndexer(config.order, contextFeaturizer).filter(contextFeatureCounts(_) > 5))
    val featIndex =
      if (config.useHashing)
        HashingFeatureIndex(positiveFeatIndex)
      else
        positiveFeatIndex

    experiment.putDisk('ContextFeaturizer, contextFeaturizer)
    experiment.putDisk('ContextFeatureIndex, featIndex)
    experiment.putDisk('PredictionFeaturizer, predictionFeaturizer)
    experiment.putDisk('VocabIndex, corpus.vocabularyIndex)

    val counts = Counter[String,Double](corpus.nGramIterator(1).map(_(0) -> 1d))
    val noiseDistribution = Multinomial[Counter[String,Double],String](counts)
    val uniformDistribution = Multinomial[Counter[String,Double],String](Counter(counts.keySet.map(_ -> 1d)))

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

          val batchNGrams = lines.flatMap { line => line.split(" ").toIndexedSeq.nGrams(config.order) }

          val contextFeatures = batchNGrams map { contextFeaturizer(_).flatMap(featIndex.indexOpt).toArray }

          val (predictionFeatures, predictionProbs) = (batchNGrams map { ngram: IndexedSeq[String] =>
            val prediction = ngram.last
            val predFeats = predictionFeaturizer(ngram)
            // for now we are assuming these are the same thing (see comment above)
            assert(predFeats.size == 1 && predFeats.last == prediction)
            (predFeats map corpus.vocabularyIndex, noiseDistribution.probabilityOf(prediction))
          }).unzip

          val (noiseFeatures, noiseProbs) = (batchNGrams map { ngram: IndexedSeq[String] =>
            val samples = if (config.objectiveType == NCE)
              noiseDistribution.sample(config.noiseSamples)
            else
              uniformDistribution.sample(config.noiseSamples)
            (samples map { sample =>
              val predFeats = predictionFeaturizer(IndexedSeq(sample))
              (predFeats map corpus.vocabularyIndex, noiseDistribution.probabilityOf(sample))
            }).unzip
          }).unzip

          val wordIds: Seq[Int] = batchNGrams map { ngram: IndexedSeq[String] => corpus.vocabularyIndex(ngram.last) }

          experiment.putDisk(Symbol(s"ContextFeatures$group"), contextFeatures)
          experiment.putDisk(Symbol(s"PredictionFeatures$group"), predictionFeatures)
          experiment.putDisk(Symbol(s"NoiseFeatures$group"), noiseFeatures)
          experiment.putDisk(Symbol(s"PredictionProbs$group"), predictionProbs)
          experiment.putDisk(Symbol(s"NoiseProbs$group"), noiseProbs)
          experiment.putDisk(Symbol(s"WordIds$group"), wordIds)
        }
      }
    }

    experiment.putDisk('NLineGroups, Int.box(corpus.lineGroupIterator(config.featureGroupSize).length))
  }
}
