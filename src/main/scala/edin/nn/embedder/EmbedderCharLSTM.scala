package edin.nn.embedder

import edin.nn.layers.Dropout
import edin.nn.model.{Any2Int, String2Int, YamlConfig}
import edu.cmu.dynet.{Expression, ParameterCollection}

sealed case class EmbedderCharLSTMConfig(
                                          w2i     : Any2Int[String],
                                          outDim  : Int,
                                          dropout : Float
                                        ) extends EmbedderConfig[String]{
  def construct()(implicit model: ParameterCollection) = new EmbedderCharLSTM(this)
}

object EmbedderCharLSTMConfig {

  def fromYaml(conf:YamlConfig) : EmbedderConfig[String] =
    EmbedderCharLSTMConfig(
      w2i     = conf("w2i").any2int,
      outDim  = conf("out-dim").int,
      dropout = conf.getOrElse("dropout", 0f)
    )

}

class EmbedderCharLSTM(config: EmbedderCharLSTMConfig)(implicit model: ParameterCollection) extends Embedder[String] {

  override val outDim: Int = config.outDim

  private val c2i = new String2Int(minCount = 10)
  for(word <- config.w2i.UNK_str::config.w2i.all_non_UNK_values)
    for(c <- word.toCharArray)
      c2i.addToCounts(c.toString)

  private val seqEmb = EmbedderSingleVectorForSeqConfig[String, String](
    outDim        = config.outDim,
    breakToPieces = {s:String => s.toCharArray.toList.map(_.toString)},
    piece2Int     = c2i
  ).construct()

  private val drop = Dropout(config.dropout)

  override def apply(xs: List[String]): Expression =
    drop(seqEmb(xs))

}
