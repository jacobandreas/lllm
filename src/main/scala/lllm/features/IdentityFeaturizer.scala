package lllm.features

import breeze.util.Index

/**
 * @author jda
 */
case class IdentityFeaturizer() extends Featurizer {

  override def apply(arg: IndexedSeq[String]): Iterable[String] = Seq(arg.last)

}
