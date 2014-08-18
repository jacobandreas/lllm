package lllm.main

import breeze.optimize.FirstOrderMinimizer.OptParams
import igor.experiment.Experiment
import lllm.model.{Hierarchical, NCE, Objective, Exact}

/**
 * @author jda
 */

case class LLLMParams(
  trainPath: String,
  testPath: String,
  order: Int = 4,
  objective: Objective = Hierarchical,
  featureGroupSize: Int = 1000,
  useHashing: Boolean = false,
  noiseSamples: Int = 100,
  rank: Int = 20,
  rareWordThreshold: Int = 10,
  rareSuffixThreshold: Int = 2,
  cacheFeatures: Boolean = false,
  nLines: Int = Integer.MAX_VALUE
)

object LLLMExperiment extends Experiment[LLLMParams] {

  protected val paramManifest = manifest[LLLMParams]
  //val stages = Seq(PrecomputeFeatures, Evaluate)
  val stages = Seq(PrecomputeFeatures, TrainModel, Evaluate)

}
