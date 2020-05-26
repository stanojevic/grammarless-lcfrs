package edin.nn.layers

import edu.cmu.dynet.{Expression, ParameterCollection, ParameterInit}
import edin.nn.DyFunctions._

sealed case class LayerNormConfig( dim:Int ){
  def construct()(implicit model: ParameterCollection) = new LayerNorm(this)
}

object LayerNorm{
  def apply(dim:Int)(implicit model: ParameterCollection) : LayerNorm = LayerNormConfig(dim).construct()
}

class LayerNorm(config:LayerNormConfig)(implicit model: ParameterCollection) extends Layer {

  private val b = addParameters(config.dim, initAround(0))
  private val g = addParameters(config.dim, initConst(1) )

  override def apply(x: Expression, targets: List[Int]): Expression = {
    if(targets.nonEmpty)
      sys.error("Layer norm not supported, yet, with subselection")
    Expression.layerNorm(x, g, b)
  }

}
