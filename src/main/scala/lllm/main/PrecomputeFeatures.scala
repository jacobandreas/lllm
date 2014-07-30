package lllm.main

import breeze.linalg.Counter
import breeze.stats.distributions.Multinomial
//import lllm.util.Multinomial
import breeze.util.Index
import erector.corpus.TextCorpusReader
import erector.util.text.toNGramIterable
import igor.experiment.{ResultCache, Stage}
import lllm.features._
import lllm.model.{NCE, Hierarchical, NGramLanguageModel}
import lllm.util.{PreprocessingIndex, RareWordPreprocessor, HuffmanDict}
import lllm.util.RichClasses.IndexWithOOV

/**
 * @author jda
 */
object PrecomputeFeatures extends Stage[LLLMParams] {

  override def run(config: LLLMParams, cache: ResultCache): Unit = {

    val corpus = TextCorpusReader(config.trainPath)
    val preCounts = Counter[String,Double](corpus.nGramIterator(1).map(_(0) -> 1d))
    val wordPreprocessor = RareWordPreprocessor(preCounts, config.rareWordThreshold, UnknownWordToken)
    val vocabIndex = PreprocessingIndex(wordPreprocessor)(corpus.vocabularyIndex)
    val counts = Counter[String,Double](corpus.nGramIterator(1).map(ngram => wordPreprocessor(ngram(0)) -> 1d))

    val contextFeaturizer = NGramFeaturizer(2, config.order) + WordAndIndexFeaturizer() + ConstantFeaturizer()
    //val contextFeaturizer = NGramFeaturizer(2, config.order) + ConstantFeaturizer()
    val predictionFeaturizer = IdentityFeaturizer(wordPreprocessor)
    // we assume for now that the features that come out of predictionFeaturizer look _exactly like_ the keys in the
    // vocabulary index above

    val contextFeatureCounts = Counter(corpus.nGramIterator(config.order).flatMap { nGram => contextFeaturizer(nGram).map((_,1)) })
    val positiveFeatIndex = Index(corpus.nGramFeatureIndexer(config.order, contextFeaturizer).filter(contextFeatureCounts(_) > 5))
    val featIndex = positiveFeatIndex

    cache.putDisk('ContextFeaturizer, contextFeaturizer)
    cache.putDisk('ContextFeatureIndex, featIndex)
    cache.putDisk('PredictionFeaturizer, predictionFeaturizer)
    cache.putDisk('VocabIndex, vocabIndex)

    val cpIndexBuilder = new CrossProductIndex.Builder(vocabIndex, featIndex, hashFeatures = if(config.useHashing) 1.0 else 0.0)

    val noiseDistribution = Multinomial(counts)
    val uniformDistribution = Multinomial(Counter(counts.keySet.map(_ -> 1d)))
    val samplingDistribution = if (config.objective == NCE)
      noiseDistribution
    else
      uniformDistribution

    if (config.objective == Hierarchical) {
      val huffmanDict = task("building Huffman dictionary") {
        counts.keysIterator.foreach { key => corpus.vocabularyIndex(key)}
        val intCounts = counts.keysIterator.map { key => (corpus.vocabularyIndex(key), counts(key))}.toIterable
        HuffmanDict.fromCounts(intCounts)
      }
      cache.putDisk('HuffmanDict, huffmanDict)
    }

    val ngramCounts = Counter(corpus.nGramIterator(config.order).map(_.map(wordPreprocessor) -> 1d))
    val ngramModel = new NGramLanguageModel(ngramCounts, wordPreprocessor)
    cache.putDisk('NGramModel, ngramModel)

    task("caching gold features") {
      val lineGroups = corpus.lineGroupIterator(config.featureGroupSize)
      lineGroups.zipWithIndex.foreach { case (lines, group) =>

        task(s"batch $group") {

          val batchNGrams = lines.flatMap { line => line.split(" ").toIndexedSeq.nGrams(config.order) }

          val contextFeatures = batchNGrams map { contextFeaturizer(_).flatMap(featIndex.indexOpt).toArray }

          val (predictionFeatures, predictionProbs) = (batchNGrams zip contextFeatures map { case (ngram, context) =>
            val prediction = wordPreprocessor(ngram.last)
            val predFeats = predictionFeaturizer(ngram)
            // for now we are assuming these are the same thing (see comment above)
            assert(predFeats.size == 1 && predFeats.last == prediction)
            val crossFeats = cpIndexBuilder.add((predFeats map vocabIndex).toArray, context)
            (crossFeats, samplingDistribution.logProbabilityOf(prediction))
          }).unzip

          val wordIds: Seq[Int] = batchNGrams map { ngram: IndexedSeq[String] => vocabIndex(ngram.last) }

          cache.putDisk(Symbol(s"ContextFeatures$group"), contextFeatures)
          cache.putDisk(Symbol(s"PredictionFeatures$group"), predictionFeatures)
          cache.putDisk(Symbol(s"PredictionProbs$group"), predictionProbs)
          cache.putDisk(Symbol(s"WordIds$group"), wordIds)
        }
      }
    }


    // we might use it below or not (if buildGuessFeatures)
    lazy val cpIndex = cpIndexBuilder.result()
    val buildGuessFeatures = false

    task("caching guess features") {
      val lineGroups = corpus.lineGroupIterator(config.featureGroupSize)
      lineGroups.zipWithIndex.foreach { case (lines, group) =>

        task(s"batch $group") {

          val batchNGrams = lines.flatMap { line => line.split(" ").toIndexedSeq.nGrams(config.order) }

          val contextFeatures = batchNGrams map { contextFeaturizer(_).flatMap(featIndex.indexOpt).toArray }

          val (noiseFeatures, noiseProbs) = (batchNGrams zip contextFeatures map { case (ngram, context) =>
            val samples = samplingDistribution.sample(config.noiseSamples)
            (samples map { preSample =>
              val sample = wordPreprocessor(preSample)
              val predFeats = predictionFeaturizer(IndexedSeq(sample))
              val crossFeats =
                if (buildGuessFeatures)
                  cpIndexBuilder.add((predFeats map vocabIndex).toArray, context)
                else
                  cpIndex.crossProduct((predFeats map vocabIndex).toArray, context)
              (crossFeats, samplingDistribution.logProbabilityOf(sample))
            }).unzip
          }).unzip

          cache.putDisk(Symbol(s"NoiseFeatures$group"), noiseFeatures)
          cache.putDisk(Symbol(s"NoiseProbs$group"), noiseProbs)
        }
      }
    }

    cache.putDisk('CrossIndex, cpIndex)
    cache.putDisk('NLineGroups, Int.box(corpus.lineGroupIterator(config.featureGroupSize).length))
  }
}
