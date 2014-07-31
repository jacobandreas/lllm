package lllm.features

/**
 * @author jda
 */
trait WordPreprocessor extends (String => String) {
  def before(featurizer: Featurizer) = new Featurizer {
    override def apply(arg: IndexedSeq[String]) = featurizer(arg map WordPreprocessor.this)
  }
}
