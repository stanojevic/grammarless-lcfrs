package edin.nn.layers

import edu.cmu.dynet.{Expression, ParameterCollection, ParameterInit}
import edin.nn.DyFunctions._

sealed case class BiasLayerConfig( dim:Int ){
  def construct()(implicit model: ParameterCollection) = new BiasLayer(this)
}

class BiasLayer(config:BiasLayerConfig)(implicit model: ParameterCollection) extends Layer {

  // something is weird about initialization
  private val b = addParameters(config.dim, initAround(0))
//  private val b    = addParameters(config.dim, scale = 0.00000001f)

  override def apply(x: Expression, targets: List[Int]): Expression = {
    if(targets.nonEmpty)
      sys.error("Bias layer is not supported, yet, with subselection")
    x+b
  }

}
