package lllm.features

import breeze.util.Index

/**
 * @author jda
 */
case class IdentityFeaturizer(preprocessor: String => String) extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Iterable[String] = Seq(preprocessor(arg.last))

}
