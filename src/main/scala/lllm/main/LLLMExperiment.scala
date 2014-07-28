package lllm.main

import breeze.optimize.FirstOrderMinimizer.OptParams
import igor.experiment.Experiment
import lllm.model.ObjectiveType._

/**
 * @author jda
 */

case class LLLMParams(
  trainPath: String,
  testPath: String,
  order: Int = 3,
  objectiveType: ObjectiveType = CD,
  featureGroupSize: Int = 1000,
  useHashing: Boolean = false,
  noiseSamples: Int = 10,
  rank: Int = 20,
  rareWordThreshold: Int = 2,
  optParams: OptParams = OptParams(useStochastic = false, batchSize = 5, maxIterations = 0)

)

object LLLMExperiment extends Experiment[LLLMParams] {

  protected val paramManifest = manifest[LLLMParams]
  val stages = Seq(PrecomputeFeatures, TrainModel, Evaluate)

}
