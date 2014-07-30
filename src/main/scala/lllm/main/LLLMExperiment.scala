package lllm.main

import breeze.optimize.FirstOrderMinimizer.OptParams
import igor.experiment.Experiment
import lllm.model.{NCE, Objective, Exact}

/**
 * @author jda
 */

case class LLLMParams(
  trainPath: String,
  testPath: String,
  order: Int = 3,
  objective: Objective = NCE,
  featureGroupSize: Int = 1000,
  useHashing: Boolean = false,
  noiseSamples: Int = 10,
  rank: Int = 20,
  rareWordThreshold: Int = 2,
  optParams: OptParams = OptParams(useStochastic = true, batchSize = 5, maxIterations = 100)

)

object LLLMExperiment extends Experiment[LLLMParams] {

  protected val paramManifest = manifest[LLLMParams]
  val stages = Seq(PrecomputeFeatures, TrainModel, Evaluate)

}
