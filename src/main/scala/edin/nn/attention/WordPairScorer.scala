package edin.nn.attention

import edin.nn.layers.MLPConfig
import edin.nn.DyFunctions._
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

import scala.collection.mutable.ArrayBuffer

trait WordPairScorer {

  val labels : Int

  //  L>1  (dim, seqLenX), b               (seqLenY, L, seqLenX), b
  //  L=1  (dim, seqLenX), b               (seqLenY, seqLenX), b
  // for cases where input is matrix with column per each word
  def apply(x:Expression, y:Expression) : Expression

  // for cases when we have a seq of vectors
  def apply(xs:List[Expression], ys:List[Expression]) : Expression

}

final case class WordPairScorerConfig(
                                      ttype     : String,
                                      dim       : Int,
                                      labels    : Int,
                                    ) {
  def construct()(implicit model:ParameterCollection): WordPairScorer = ttype match {
    case "biaffine" => new WordPairScorerBiAffine(this)
    case "mlp" => new WordPairScorerMLP(this)
  }
}

object WordPairScorerConfig{

  def fromYaml(conf:YamlConfig) : WordPairScorerConfig =
    WordPairScorerConfig(
      ttype = conf("type").str,
      dim = conf("dim").int,
      labels = conf("labels").int
    )

}


