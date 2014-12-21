package lllm.main

import igor.experiment.{Stage, Experiment}

/**
 * @author jda
 */

case class SelfNormalizingParams(
  trainPath: String,
  order: Int = 3,
  lineGroupSize: Int = 100,
  numLines: Int = 0,
  numEpochs: Int = 3,
  minFeatureCount: Int = 5,
  minSuffixCount: Int = 20,
  minWordCount: Int = 25,
  useNGramFeatures: Boolean = true,
  useSkipGramFeatures: Boolean = true,
  nGramMinOrder: Int = 2,
  nGramMaxOrder: Int = 3,
  paramL2: Double,
  partitionL2: Double,
  partitionFrac: Double
)

object SelfNormalizingExperiment extends Experiment[SelfNormalizingParams] {
  protected val paramManifest = manifest[SelfNormalizingParams]
  override def stages = Seq(PrepFeaturesSN, TrainSN)
}
