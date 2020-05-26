package edin.nn.set

import edin.nn.layers.{IdentityLayer, Layer, SingleLayerConfig}
import edin.nn.DyFunctions._
import edin.nn.model.YamlConfig
import edin.nn.{DyFunctions, State, StateClosed}
import edu.cmu.dynet.{Expression, ParameterCollection}

sealed case class DeepSetConfig(
                                 inDim            : Int,
                                 outDim           : Int,
                                 dropout          : Float,
                                 withPreEncoding  : Boolean,
                                 withPostEncoding : Boolean,
                                 withLayerNorm    : Boolean,
                                 midDim           : Int,
                                 activation       : String
                               ){

  def construct()(implicit model: ParameterCollection) = new DeepSet(this)

}

object DeepSetConfig{

  def fromYaml(conf:YamlConfig) : DeepSetConfig = {

    val withPreEncoding = conf.getOrElse("with-pre-encoding", true)

    val withPostEncoding = conf.getOrElse("with-post-encoding", true)

    val midDim = if(conf.contains("mid-dim"))
      conf("mid-dim").int
    else if(! withPreEncoding)
      conf("in-dim").int
    else if(! withPostEncoding)
      conf("out-dim").int
    else
      throw new Exception("I don't know what to do in this situation")

    DeepSetConfig(
      inDim            = conf("in-dim"     ).int,
      outDim           = conf("out-dim"    ).int,
      midDim           = midDim,
      activation       = conf("activation" ).str,
      withLayerNorm    = conf.getOrElse("with-layer-norm", false),
      dropout          = conf.getOrElse("dropout", 0f),
      withPreEncoding  = conf("with-pre-encoding"  ).bool,
      withPostEncoding = conf("with-post-encoding" ).bool
    )

  }

}

final class DeepSet(config:DeepSetConfig)(implicit model: ParameterCollection) {

  assert(config.withPreEncoding  || config.withPostEncoding        )
  assert(config.withPreEncoding  || config.midDim == config.inDim  )
  assert(config.withPostEncoding || config.midDim == config.outDim )

  private val phi : Layer = if(config.withPreEncoding)
    SingleLayerConfig(
      inDim = config.inDim,
      outDim = config.midDim,
      activationName = "linear",
      withLayerNorm = config.withLayerNorm,
      dropout = config.dropout
    ).construct()
  else
    IdentityLayer()


  private val rho : Layer = if(config.withPostEncoding)
    SingleLayerConfig(
      inDim = config.midDim,
      outDim = config.outDim,
      activationName = config.activation,
      withLayerNorm = config.withLayerNorm,
      dropout = config.dropout
    ).construct()
  else
    IdentityLayer()

  def preEncode(element: Expression): Expression = phi(element)

  def postEncode(midEncode: Expression): Expression = rho(midEncode)

  def postEncode(midEncodes: List[Expression]): Expression =
    if(midEncodes.isEmpty)
      postEncode(zerosMidEncoding)
    else
      postEncode(esum(midEncodes))

  def encodeSet(set:List[Expression]) : Expression =
    postEncode(set map preEncode)

  // def encodeWeightedSet(wset:List[(Expression, Float)]) : Expression = encodeSet(wset.map(x => x._1*x._2))

  // def initStateUnTyped() : DeepSetStateSimplified = new DeepSetStateSimplified(this.zeros, this)

  private def initStateTyped[T<:State]() : DeepSetStateTyped[T] = new DeepSetStateTyped(this.zerosMidEncoding, Set(), this)

  def empty[T<:State]() : DeepSetStateTyped[T] = initStateTyped()

  private def zerosMidEncoding : Expression = DyFunctions.zeros(config.inDim)

}

//final class DeepSetStateSimplified(midEncoding:Expression, ds:DeepSet) extends State {
//
//  lazy val h: Expression = ds.postEncode(midEncoding)
//
//  def add(el:Expression) : DeepSetStateSimplified = new DeepSetStateSimplified(midEncoding + ds.preEncode(el), ds)
//  def add(els:List[Expression]) : DeepSetStateSimplified = els.foldLeft(this){case (state, el) => state.add(el)}
//
//  def remove(el:Expression) : DeepSetStateSimplified = new DeepSetStateSimplified(midEncoding - ds.preEncode(el), ds)
//  def remove(els:List[Expression]) : DeepSetStateSimplified = els.foldLeft(this){case (state, el) => state.remove(el)}
//
//}

final class DeepSetStateTyped[T <: State](midEncoding: => Expression, val elems:Set[T], ds:DeepSet) extends StateClosed {

  lazy val h: Expression = ds.postEncode(midEncoding)

  def +(el:T) : DeepSetStateTyped[T] =
    if(elems contains el)
      this
    else
      new DeepSetStateTyped(midEncoding + ds.preEncode(el.h), elems+el, ds)

  def +(els:List[T]) : DeepSetStateTyped[T] =
    els.foldLeft(this){case (state, el) => state + el}

  def -(el:T) : DeepSetStateTyped[T] =
    if(elems contains el)
      this
    else
      new DeepSetStateTyped(midEncoding - ds.preEncode(el.h), elems-el, ds)

  def -(els:List[T]) : DeepSetStateTyped[T] =
    els.foldLeft(this){case (state, el) => state - el}

}
