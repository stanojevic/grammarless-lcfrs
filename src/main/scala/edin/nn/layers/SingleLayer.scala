package edin.nn.layers

import edin.nn.DyFunctions._
import edin.nn.model.YamlConfig
import edu.cmu.dynet._

// TODO !!! batch support for subselecting

sealed case class SingleLayerConfig(
                                     inDim          : Int,
                                     outDim         : Int,
                                     activationName : String,
                                     withLayerNorm  : Boolean,
                                     dropout        : Float,
                                     dropConnect    : Boolean = false,
                                     initializer    : String  = "glorot"
                                   ){
  def construct()(implicit model: ParameterCollection) = new SingleLayer(this)
}

object SingleLayerConfig{

  def fromYaml(conf:YamlConfig) : SingleLayerConfig =
    SingleLayerConfig(
      inDim          = conf("in-dim"         ).int,
      outDim         = conf("out-dim"        ).int,
      activationName = conf("activation"     ).str,
      withLayerNorm  = conf.getOrElse("with-layer-norm" , false),
      dropout        = conf.getOrElse("dropout", 0f),
      initializer    = conf.getOrElse("initializer", "glorot")
    )

}

class SingleLayer(config:SingleLayerConfig)(implicit model:ParameterCollection) extends Layer {

  require(!config.withLayerNorm || !config.activationName.toLowerCase.endsWith("softmax"))

  private val activation = activationFactory(config.activationName)
  private val param_W    = addParameters((config.outDim, config.inDim), init=init(config.initializer, config.inDim, config.outDim, config.activationName))
  private val param_b    = addParameters( config.outDim, initAround(0))
  private val param_g    = if(config.withLayerNorm) Some(addParameters( config.outDim, initConst(1) )) else None // for layer norm
  private val drop       = Dropout(config.dropout)

  System.err.println("creating "+this)

  override def toString: String = s"Layer(${config.inDim.toString}, ${config.outDim.toString})"

  private def activate(W:Expression, input:Expression, b:Expression, g:Option[Expression]) : Expression = g match {
    case Some(g) =>
      drop(activation(Expression.layerNorm(W*input, g, b)))
    case None =>
      drop(activation(W*input+b))
  }

  private def subSelectForTarget(targets:List[Int], W:Expression, b:Expression, g_layerNorm:Option[Expression]) : (Expression, Expression, Option[Expression]) =
    targets match {
      case Nil =>
        (W, b, g_layerNorm)
      case subselect =>
        (
          selectRows(W, subselect),
          selectRows(b, subselect),
          g_layerNorm.map(selectRows(_, subselect))
        )
    }

  def apply(input:Expression, targets:List[Int]=List()): Expression = {
    if(targets.nonEmpty && input.batchSize>1)
      sys.error("you can't do subselecting in the batch mode")
    val (w, b, g) = subSelectForTarget(targets, param_W, param_b, param_g)
    activate(w, input, b, g)
  }

}

object SingleLayer{

  def compressor(inDim: Int, outDim: Int, withLayerNorm:Boolean=false, initializer:String="glorot", dropConnect:Boolean=false)(implicit model:ParameterCollection) : Layer =
    SingleLayerConfig(
      inDim          = inDim,
      outDim         = outDim,
      activationName = "nothing",
      withLayerNorm  = withLayerNorm,
      dropout        = 0f,
      dropConnect    = dropConnect,
      initializer    = initializer
    ).construct()

}

