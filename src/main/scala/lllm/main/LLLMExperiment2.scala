package lllm.main

import igor.experiment.Experiment
import lllm.model.Regularizer

/**
 * @author jda
 */

case class LLLMParams2(
  trainPath: String,
  testPath: String,
  order: Int = 3,
  lineGroupSize: Int = 100,
//  nLines: Int = 1000,
  nLines: Int = 0,
  nEpochs: Int = 10,
  safeEval: Boolean = false,
  regularizerType: Regularizer = Regularizer.L2,
  regularizerStrength: Double = .05,
  minFeatureCount: Int = 5,
  minSuffixCount: Int = 20,
  minWordCount: Int = 100,
//  minSuffixCount: Int = 0,
//  minWordCount: Int = 0,
  useNGramFeatures: Boolean = true,
  useSkipGramFeatures: Boolean = true,
  nGramMinOrder: Int = 2,
  nGramMaxOrder: Int = 3
)

object LLLMExperiment2 extends Experiment[LLLMParams2] {
  protected val paramManifest = manifest[LLLMParams2]
  val stages = Seq(PrepFeatures2, TrainModel2, Evaluate2)
  //val stages = Seq(Evaluate2)
}

// 459
// 437
// 432

// mean rank
// bleu
