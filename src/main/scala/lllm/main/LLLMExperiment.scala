package lllm.main

import igor.experiment.{Launcher, Experiment}
import igor.config.Configuration

/**
 * @author jda
 */
class LLLMExperiment(config: Configuration) extends Experiment(config) {

  def runStages() {
    stage[PrecomputeFeatures]
    stage[TrainModel]
    stage[Evaluate]
  }

}

object LLLMExperiment extends Launcher {
  override def buildExperiment(config: Configuration) = new LLLMExperiment(config)
}
