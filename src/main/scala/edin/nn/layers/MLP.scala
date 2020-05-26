package edin.nn.layers

import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

sealed case class MLPConfig(
                           activations    : List[String]         , // n   elements
                           sizes          : List[Int]            , // n+1 elements
                           dropouts       : List[Float] = List() , // n   elements
                           withLayerNorm  : Boolean     = false
                           ){
  require(activations.size + 1 == sizes.size)
  require(dropouts.isEmpty || dropouts.size == activations.size)

  def construct()(implicit model: ParameterCollection) = new MLP(this)
}

object MLPConfig{

  def fromYaml(conf:YamlConfig) : MLPConfig =
    MLPConfig(
      activations = conf("activations").strList,
      sizes = conf("sizes").intList,
      dropouts = conf.getOptionalListFloat("dropouts"),
      withLayerNorm = conf.getOrElse("with-layer-norm", false)
    )

}

class MLP(config:MLPConfig)(implicit model: ParameterCollection) extends Layer{

  val outDim: Int = config.sizes.last

  private def zip4[A,B,C,D](as:List[A], bs:List[B], cs:List[C], ds:List[D]) : List[(A,B,C,D)] =
    (((as zip bs) zip cs) zip ds).map{ case (((a, b), c), d) => (a, b, c, d) }

  private val dropProbs = if(config.dropouts.isEmpty) List.fill[Float](config.activations.size)(0f) else config.dropouts

  private val layers = zip4(config.sizes.init, config.sizes.tail, dropProbs, config.activations).map{
    case (inDim, outDim, dropProb, activationName) =>
      SingleLayerConfig(
        inDim          = inDim,
        outDim         = outDim,
        activationName = activationName,
        withLayerNorm  = config.withLayerNorm,
        dropout        = dropProb
      ).construct()
  }

  override def toString: String =
    "MLP(" + layers.mkString(", ") + ")"

  def apply(x:Expression, targets:List[Int]=List()) : Expression = {
    val preLast = layers.init.foldLeft(x){(x, layer) => layer(x)}
    layers.last(preLast, targets)
  }

}
