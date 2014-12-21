package lllm.old.main

import igor.experiment.Experiment
import lllm.old.model.{Objective, Hierarchical}

/**
 * @author jda
 */

case class LLLMParams(
  trainPath: String,
  testPath: String,
  order: Int = 5,
  objective: Objective = Hierarchical,
  featureGroupSize: Int = 100,
  useHashing: Boolean = false,
  noiseSamples: Int = 100,
  rank: Int = 20,
  rareWordThreshold: Int = 10,
  rareSuffixThreshold: Int = 10,
  cacheFeatures: Boolean = false,
  nLines: Int = Integer.MAX_VALUE
)

object LLLMExperiment extends Experiment[LLLMParams] {

  protected val paramManifest = manifest[LLLMParams]
  //val stages = Seq(PrecomputeFeatures, Evaluate)
  val stages = Seq(PrecomputeFeatures, TrainModel, Evaluate)

}
