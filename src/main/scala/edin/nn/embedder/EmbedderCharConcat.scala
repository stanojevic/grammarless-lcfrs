package edin.nn.embedder

import edin.nn.layers.{Dropout, IdentityLayer, SingleLayer}
import edin.nn.model.{Any2Int, String2Int, YamlConfig}
import edin.nn.DyFunctions._
import edu.cmu.dynet.{Dim, Expression, ParameterCollection}

// Embedder originally proposed by
// Kitaev and Klein in "Constituency Parsing with a Self-Attentive Encoder"
// https://arxiv.org/pdf/1805.01052.pdf

sealed case class EmbedderCharConcatConfig(
                                          w2i     : Any2Int[String],
                                          outDim  : Int,
                                          dropout : Float
                                        ) extends EmbedderConfig[String]{
  def construct()(implicit model: ParameterCollection) = new EmbedderCharConcat(this)
}

object EmbedderCharConcatConfig{

  def fromYaml(conf:YamlConfig) : EmbedderConfig[String] =
    EmbedderCharLSTMConfig(
      w2i     = conf("w2i").any2int,
      outDim  = conf("out-dim").int,
      dropout = conf.getOrElse("dropout", 0f)
    )

}

class EmbedderCharConcat(config: EmbedderCharConcatConfig)(implicit model: ParameterCollection) extends Embedder[String] {

  override val outDim: Int = config.outDim

  private val compressor = if(outDim != 512) SingleLayer.compressor(512, outDim) else IdentityLayer()

  private val c2i = new String2Int(
    minCount          = 10,
    maxVacabulary     = Int.MaxValue
  )
  for(word <- config.w2i.UNK_str::config.w2i.all_non_UNK_values)
    for(c <- word.toCharArray)
      c2i.addToCounts(c.toString)

  private val E = model.addLookupParameters(c2i.size, Dim(List(32)))
  private val drop = Dropout(config.dropout)

  override def apply(x: List[String]): Expression =
    x.map{ s =>
      val is = s.toCharArray.map(c => c2i(c.toString))
      val front = is.take(8) ++ Array.fill(Math.max(0, 8-is.length))(c2i.UNK_i)
      val back  = is.reverse.take(8) ++ Array.fill(Math.max(0, 8-is.length))(c2i.UNK_i)
      front ++ back
    }.transpose.map{ vertical =>
      drop(E(vertical))
    }.econcat

}
