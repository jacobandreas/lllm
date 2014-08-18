package lllm.main

import breeze.linalg.{DenseVector, Counter}
import breeze.stats.distributions.Multinomial
//import lllm.util.Multinomial
import breeze.util.Index
import erector.corpus.TextCorpusReader
import erector.util.text.toNGramIterable
import igor.experiment.{ResultCache, Stage}
import erector.learning.Feature
import lllm.features._
import lllm.model.{KneserNeyLanguageModel, NCE, Hierarchical, NGramLanguageModel}
import lllm.util.{PreprocessingIndex, HuffmanDict}
import lllm.util.RichClasses.IndexWithOOV
import scala.io.Source

/**
 * @author jda
 */
object PrecomputeFeatures extends Stage[LLLMParams] {

  override def run(config: LLLMParams, cache: ResultCache): Unit = {

    implicit val (_config, _cache) = (config, cache)

    val corpus = TextCorpusReader(config.trainPath).dummy(config.nLines)
    val preCounts = Counter(corpus.nGramIterator(1).map(_(0) -> 1d))
    //val wordPreprocessor = RareWordPreprocessor(preCounts, config.rareWordThreshold, UnknownWordToken)
    val contextPreprocessor = FrequentSuffixPreprocessor(preCounts, config.rareSuffixThreshold, UnknownWordToken)
    //val predictionPreprocessor = RareWordPreprocessor(preCounts, config.rareWordThreshold, UnknownWordToken)
    val predictionPreprocessor = new WordPreprocessor with Serializable {
      override def apply(x: String) = x
    }
    //val contextPreprocessor = predictionPreprocessor
    val vocabIndex = PreprocessingIndex(predictionPreprocessor)(corpus.vocabularyIndex)
    val counts = Counter[String, Double](corpus.nGramIterator(1).map(ngram => predictionPreprocessor(ngram(0)) -> 1d))

//    val contextFeaturizer = contextPreprocessor before
//      (NGramFeaturizer(2, config.order) +
//       WordAndIndexFeaturizer() +
//       ConstantFeaturizer())
    val contextFeaturizer = contextPreprocessor before (NGramFeaturizer(2, config.order))
    //val contextFeaturizer = NGramFeaturizer(2, config.order) + ConstantFeaturizer()
    val predictionFeaturizer = predictionPreprocessor before IdentityFeaturizer()
    // we assume for now that the features that come out of predictionFeaturizer look _exactly like_ the keys in the
    // vocabulary index above

    val contextFeatureCounts = Counter(corpus.nGramIterator(config.order).flatMap { nGram => contextFeaturizer(nGram).map((_, 1d))})
    // TODO(jda) why doesn't this work?
    //val positiveFeatIndex = Index(corpus.nGramFeatureIndexer(config.order, contextFeaturizer).filter(contextFeatureCounts(_) > 5))
    val positiveFeatIndex = Index(corpus.nGramIterator(config.order).flatMap(contextFeaturizer.apply).filter(contextFeatureCounts(_) >= 2))
    val featIndex = positiveFeatIndex

    cache.put('ContextFeaturizer, contextFeaturizer)
    cache.put('ContextFeatureIndex, featIndex)
    cache.put('PredictionFeaturizer, predictionFeaturizer)
    cache.put('PredictionPreprocessor, predictionPreprocessor)
    cache.put('FeatIndex, featIndex)
    cache.put('VocabIndex, vocabIndex)

    val cpIndexBuilder = new CrossProductIndex.Builder(vocabIndex, featIndex, hashFeatures = if (config.useHashing) 1.0 else 0.0)

    val noiseDistribution = Multinomial(counts)
    val uniformDistribution = Multinomial(Counter(counts.keySet.map(_ -> 1d)))
    val samplingDistribution = if (config.objective == NCE)
      noiseDistribution
    else
      uniformDistribution

    cache.put('SamplingDistributionParams, samplingDistribution.params)

    val huffmanDict =
      if (config.objective == Hierarchical) {
        val hd = task("building Huffman dictionary") {
          counts.keysIterator.foreach { key => corpus.vocabularyIndex(key)}
          val intCounts = counts.keysIterator.map { key => (corpus.vocabularyIndex(key), counts(key))}.toIterable
          HuffmanDict.fromCounts(intCounts)
        }
        cache.put('HuffmanDict, hd)
        hd
      } else {
        null
      }

    val ngramCounts = (1 to config.order).map { order =>
      order -> Counter(corpus.nGramIterator(order) map { ngram =>
        //(ngram.take(ngram.length - 1).map(contextPreprocessor) :+ predictionPreprocessor(ngram.last)) -> 1d
        (ngram.take(ngram.length - 1) :+ predictionPreprocessor(ngram.last)) -> 1d
      })
    }.toMap

    val ngramModel = KneserNeyLanguageModel(ngramCounts, contextPreprocessor, predictionPreprocessor)
    cache.putDisk('NGramModel, ngramModel)
    // TODO(jda) standardize ngram casing throughout

    indexFeatures(corpus, contextFeaturizer, predictionFeaturizer, predictionPreprocessor, featIndex, vocabIndex, cpIndexBuilder, samplingDistribution, huffmanDict)
  }

  def indexFeatures(corpus: TextCorpusReader,
                    contextFeaturizer: Featurizer,
                    predictionFeaturizer: Featurizer,
                    predictionPreprocessor: WordPreprocessor,
                    featIndex: Index[Feature],
                    vocabIndex: Index[String],
                    cpIndexBuilder: CrossProductIndex.Builder,
                    samplingDistribution: Multinomial[Counter[String,Double],String],
                    huffmanDict: HuffmanDict[Int])
                   (implicit config: LLLMParams,
                             cache: ResultCache): Unit = {

    task("caching gold features") {
      val lineGroups = corpus.lineGroupIterator(config.featureGroupSize)
      lineGroups.zipWithIndex.foreach { case (lines, group) =>

        task(s"batch $group") {

          val batchNGrams = makeBatchNGrams(lines.toIterable).toArray
          val contextFeatures = makeBatchContextFeatures(batchNGrams, contextFeaturizer, featIndex)

          val (predictionFeatures, predictionProbs) = (batchNGrams zip contextFeatures map { case (ngram, context) =>
            val prediction = predictionPreprocessor(ngram.last)
            val predFeats = predictionFeaturizer(ngram)
            // for now we are assuming these are the same thing (see comment above)
            assert(predFeats.size == 1 && predFeats.last == prediction)
            val crossFeats =
              if (config.objective == Hierarchical) {
                val code = huffmanDict.dict.get(vocabIndex(prediction)).get
                //println(code)
                //println(context.mkString(","))
                // force evaluation now
                code.tails.toArray.map { path =>
                  val cpFeats = cpIndexBuilder.add(Array(huffmanDict.prefixIndex(path)),
                                                   context)
                  path -> cpFeats
                }.toMap
              } else {
                cpIndexBuilder.add((predFeats map vocabIndex).toArray, context)
              }
            (crossFeats, samplingDistribution.logProbabilityOf(prediction))
          }).unzip

          val wordIds: Seq[Int] = batchNGrams.map { ngram: IndexedSeq[String] => vocabIndex(ngram.last)}.toSeq

          if (config.cacheFeatures) {
            cache.putDisk(Symbol(s"ContextFeatures$group"), contextFeatures)
            cache.putDisk(Symbol(s"PredictionFeatures$group"), predictionFeatures)
            cache.putDisk(Symbol(s"PredictionProbs$group"), predictionProbs)
            cache.putDisk(Symbol(s"WordIds$group"), wordIds)
          } else {
            //cache.putDisk(Symbol(s"Lines$group"), lines)
            cache.writeFile(Symbol(s"Lines$group"), lines.mkString("\n"))
          }
        }
      }
    }

    // we might use it below or not (if buildGuessFeatures)
    lazy val cpIndex = cpIndexBuilder.result()
    val buildGuessFeatures = false

    if (config.cacheFeatures || buildGuessFeatures) {
      task("caching guess features") {
        val lineGroups = corpus.lineGroupIterator(config.featureGroupSize)
        lineGroups.zipWithIndex.foreach { case (lines, group) =>

          task(s"batch $group") {

            val batchNGrams = makeBatchNGrams(lines.toIterable)
            val contextFeatures = makeBatchContextFeatures(batchNGrams, contextFeaturizer, featIndex)

            val (noiseFeatures, noiseProbs) = (batchNGrams zip contextFeatures map { case (ngram, context) =>
              val samples = samplingDistribution.sample(config.noiseSamples)
              (samples map { preSample =>
                val sample = predictionPreprocessor(preSample)
                val predFeats = predictionFeaturizer(IndexedSeq(sample))
                val crossFeats =
                  if (buildGuessFeatures)
                    cpIndexBuilder.add((predFeats map vocabIndex).toArray, context)
                  else
                    cpIndex.crossProduct((predFeats map vocabIndex).toArray, context)
                (crossFeats, samplingDistribution.logProbabilityOf(sample))
              }).unzip
            }).unzip

            if (config.cacheFeatures) {
              cache.putDisk(Symbol(s"NoiseFeatures$group"), noiseFeatures)
              cache.putDisk(Symbol(s"NoiseProbs$group"), noiseProbs)
            }

          }
        }
      }
    }

    cache.put('CrossIndex, cpIndex)
    cache.put('NLineGroups, Int.box(corpus.lineGroupIterator(config.featureGroupSize).length))
  }

  def makeBatchNGrams(lines: Iterable[String])(implicit config: LLLMParams): Iterable[IndexedSeq[String]] = {
    lines.flatMap { line => line.split(" ").toIndexedSeq.nGrams(config.order)}
  }

  def makeBatchContextFeatures(batchNGrams: Iterable[IndexedSeq[String]],
                               contextFeaturizer: Featurizer,
                               featIndex: Index[Feature]): Iterable[Array[Int]] = {
    batchNGrams map {
      contextFeaturizer(_).flatMap(featIndex.indexOpt).toArray
    }
  }

  // TODO(jda) refactor

  def recomputeBatchDataFeatures(lines: Iterable[String],
                                 contextFeaturizer: Featurizer,
                                 predictionFeaturizer: Featurizer,
                                 predictionPreprocessor: WordPreprocessor,
                                 featIndex: Index[Feature],
                                 vocabIndex: Index[String],
                                 cpIndex: CrossProductIndex,
                                 samplingDistribution: Multinomial[Counter[String,Double],String])
                                (implicit config: LLLMParams,
                                          cache: ResultCache): (Iterable[Array[Int]], Iterable[Double]) = {
    val batchNGrams = makeBatchNGrams(lines)
    val contextFeatures = makeBatchContextFeatures(batchNGrams, contextFeaturizer, featIndex)
    (batchNGrams zip contextFeatures map { case (ngram, context) =>
      val prediction = predictionPreprocessor(ngram.last)
      val predFeats = predictionFeaturizer(ngram)
      // for now we are assuming these are the same thing (see comment above)
      assert(predFeats.size == 1 && predFeats.last == prediction)
      val crossFeats = cpIndex.crossProduct((predFeats map vocabIndex).toArray, context)
      (crossFeats, samplingDistribution.logProbabilityOf(prediction))
    }).unzip
  }

  def recomputeBatchNoiseFeatures(lines: Iterable[String],
                                  contextFeaturizer: Featurizer,
                                  predictionFeaturizer: Featurizer,
                                  predictionPreprocessor: WordPreprocessor,
                                  featIndex: Index[Feature],
                                  vocabIndex: Index[String],
                                  cpIndex: CrossProductIndex,
                                  samplingDistribution: Multinomial[Counter[String,Double],String])
                                 (implicit config: LLLMParams,
                                           cache: ResultCache): (Iterable[IndexedSeq[Array[Int]]], Iterable[IndexedSeq[Double]]) = {
    val batchNGrams = makeBatchNGrams(lines)
    val contextFeatures = makeBatchContextFeatures(batchNGrams, contextFeaturizer, featIndex)

    (batchNGrams zip contextFeatures map { case (ngram, context) =>
      val samples = samplingDistribution.sample(config.noiseSamples)
      (samples map { preSample =>
        val sample = predictionPreprocessor(preSample)
        val predFeats = predictionFeaturizer(IndexedSeq(sample))
        val crossFeats = cpIndex.crossProduct((predFeats map vocabIndex).toArray, context)
        (crossFeats, samplingDistribution.logProbabilityOf(sample))
      }).unzip
    }).unzip
  }

}
