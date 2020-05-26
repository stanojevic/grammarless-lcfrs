package edin.nn.embedder

import edin.nn.DyFunctions._
import edin.algorithms.MathArray
import edin.nn.layers.Dropout
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

final case class EmbedderPositionsSinusoidConfig(
                                                  outDim      : Int,
                                                  dropout     : Float=0f
                                                ) extends EmbedderConfig[Int] {
  def construct()(implicit model: ParameterCollection) = new EmbedderPositionsSinusoid(this)
}

object EmbedderPositionsSinusoidConfig{

  def fromYaml[Int](conf:YamlConfig) : EmbedderConfig[Int] =
    EmbedderPositionsSinusoidConfig(
      outDim      = conf("out-dim").int,
      dropout     = conf.getOrElse("dropout", 0f)
    ).asInstanceOf[EmbedderConfig[Int]]

}

class EmbedderPositionsSinusoid(config:EmbedderPositionsSinusoidConfig)(implicit model: ParameterCollection) extends Embedder[Int] {

  private val drop = Dropout(config.dropout)

  override val outDim: Int = config.outDim

  private val proVec:MathArray = MathArray((0 until outDim).map{i : Int => math.pow(10000.0, -i.toDouble*2/outDim).toFloat}.toArray)

  private def ineffective:Boolean = outDim == 0

  private def getEmbArray(pos:Int) : Array[Float] =
    proVec.map(x =>
      if(x %2 == 0)
        Math.sin(pos*x)
      else
        Math.cos(pos*x)
    ).array

  override def apply(xs: List[Int]): Expression =
    if(ineffective) null
    else drop(batchVector(xs map getEmbArray))

}
