package edin.nn.embedder

import edin.nn.DyFunctions._
import edin.nn.layers.Dropout
import edin.nn.model.YamlConfig
import edu.cmu.dynet.LookupParameter
import edu.cmu.dynet.{Expression, ParameterCollection}

sealed case class EmbedderPositionsLearnedConfig(
                                             maxPosition : Int,
                                             outDim      : Int,
                                             dropout     : Float=0f
                                           ) extends EmbedderConfig[Int] {
  def construct()(implicit model: ParameterCollection) = new EmbedderPositionsLearned(this)
}

object EmbedderPositionsLearnedConfig{

  def fromYaml[Int](conf:YamlConfig) : EmbedderConfig[Int] =
    EmbedderPositionsLearnedConfig(
      maxPosition = conf("max-position").int,
      outDim      = conf("out-dim").int,
      dropout     = conf.getOrElse("dropout", 0f)
    ).asInstanceOf[EmbedderConfig[Int]]

}

class EmbedderPositionsLearned(config:EmbedderPositionsLearnedConfig)(implicit model: ParameterCollection) extends Embedder[Int] {

  override val outDim: Int = config.outDim

  private val drop        = Dropout(config.dropout)
  private val maxPosition = config.maxPosition
  private val E:LookupParameter = model.addLookupParameters(config.maxPosition+1, config.outDim)

  private def ineffective:Boolean = maxPosition == 0 || outDim == 0

//  def apply(pos:Int) : Expression =
//    if(ineffective) null
//    else drop(E(clip(pos)))

  @inline private def clip(pos:Int) : Int =
    if(pos < maxPosition) pos else maxPosition

  override def apply(xs: List[Int]): Expression =
    if(ineffective) null
    else drop(E(xs map clip))

}
