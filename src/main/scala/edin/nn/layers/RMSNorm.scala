package edin.nn.layers

import edu.cmu.dynet.{Expression, ParameterCollection, ParameterInit}
import edin.nn.DyFunctions._

sealed case class RMSNormConfig( dim:Int ){
  def construct()(implicit model: ParameterCollection) = new RMSNorm(this)
}

class RMSNorm(config:RMSNormConfig)(implicit model: ParameterCollection) extends Layer {

  private val b = addParameters(config.dim, initAround(0))
  private val g = addParameters(config.dim, initConst(1) )
  private val denominator = math.pow(config.dim, -0.5)

  override def apply(x: Expression, targets: List[Int]): Expression = {
    if(targets.nonEmpty)
      sys.error("RMS norm not supported, yet, with subselection")
    val rms  = scalar(denominator)*Expression.l2Norm(x)+Ɛ
    val renormalizedInput = x/rms
    g.exp ⊙ renormalizedInput + b.exp
  }

}
