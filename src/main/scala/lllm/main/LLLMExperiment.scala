package lllm.main

import breeze.config.ArgumentParser
import igor.experiment.{Stages,Experiment}
import igor.config.Configuration
import lllm.model.ObjectiveType._

/**
 * @author jda
 */

case class LLLMParams(
  trainPath: String,
  testPath: String,
  order: Int = 3,
  objectiveType: ObjectiveType = LowRank,
  featureGroupSize: Int = 1000,
  useHashing: Boolean = false,
  noiseSamples: Int = 10,
  rank: Int = 20
)

object LLLMExperiment extends Stages[LLLMParams] {

  protected val paramManifest = manifest[LLLMParams]

  def run(exp: Experiment[LLLMParams]) {
    exp.stage(PrecomputeFeatures)
    exp.stage(TrainModel)
    exp.stage(Evaluate)
  }

}
