package lllm.model

import java.io.File

import breeze.numerics.pow
import breeze.util.Index
import erector.corpus.TextCorpusReader

import scala.io.Source
import scala.sys.process._
import scala.sys.process

/**
 * @author jda
 */
class KenLMWrapper extends LanguageModel {

  override def prob(ngram: IndexedSeq[String]): Double = ???

  override def logProb(ngram: IndexedSeq[String]): Double = ???

}

object KenLMPerplexityEvaluator {

  def apply(path: String): Double = {
    val vocab = makeVocabIndex()

    val pb = Process("/Users/jda/Code/3p/kenlm/bin/query /Users/jda/Code/3p/kenlm/euro.kenlm") #< new File(path)
    //val pb = "echo I will join the board" #> "/Users/jda/Code/3p/kenlm/bin/query /Users/jda/Code/lllm/wsj.kenlm"

    var logProb = 0d
    var count = 0d

    pb.lineStream.dropRight(3)foreach { line =>
      //println(line)
      val parts = line.trim.split("\t")
      parts.dropRight(1).foreach { part =>
        val word = part.split("=")(0)
        val lp = part.split(" ").last.toDouble
        if (vocab contains word) {
          logProb += lp
          count += 1
          //println(word, lp)
        }
      }
    }

    pow(10d, -logProb / count)

//    val pio = new ProcessIO(inputWriter, outputReader, _ => ())
//    val pb = Process("/Users/jda/Code/3p/kenlm/bin/query")
//    pb.run(pio)

  }

  def makeVocabIndex(): Index[String] = {
    val src = Source.fromFile("/Users/jda/Code/lllm/Vocab")
    Index[String] {
      src.getLines().map { line => line.stripLineEnd }
    }
  }
}
