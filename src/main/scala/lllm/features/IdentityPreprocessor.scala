package lllm.features

/**
 * @author jda
 */
object IdentityPreprocessor extends WordPreprocessor {
  override def apply(v1: String): String = v1
}
