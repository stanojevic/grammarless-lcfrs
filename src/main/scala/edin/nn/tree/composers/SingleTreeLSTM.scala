package edin.nn.tree.composers

import edin.nn.{StateClosed, State}
import edin.nn.DyFunctions._
import edin.nn.layers.{Layer, SingleLayer}
import edu.cmu.dynet.{Expression, ParameterCollection}

sealed case class SingleTreeLSTMConfig(
                                      inDim          : Int, // dimension of the node?
                                      outDim         : Int, // dimension of the computed representation?
                                      maxArity       : Int,
                                      withLayerNorm  : Boolean,
                                      dropout        : Float
                                    ) extends CompositionFunctionConfig {
  def construct()(implicit model: ParameterCollection) = new SingleTreeLSTM(this)
}

final class SingleTreeLSTM(config:SingleTreeLSTMConfig)(implicit model: ParameterCollection) extends CompositionFunction {

  if(config.withLayerNorm)
    System.err.println("WARNING 'LAYER NORM' NOT SUPPORTED IN TREE-LSTM")

  // this is implementation of "N-ary Tree-LSTM" version of Tree-LSTM from Tai. et al. paper
  private type Activation = Expression => Expression

  private val Wi  : SingularParameter             = addParameters((config.outDim,config.inDim))
  private val Ui  : List[SingularParameter]       = generateParamsList(config.outDim, config.outDim, config.maxArity)
  private val bi  : SingularParameter             = addParameters(config.outDim)

  private val Wo  : SingularParameter             = addParameters((config.outDim,config.inDim))
  private val Uo  : List[SingularParameter]       = generateParamsList(config.outDim, config.outDim, config.maxArity)
  private val bo  : SingularParameter             = addParameters(config.outDim)

  private val Wu  : SingularParameter             = addParameters((config.outDim,config.inDim))
  private val Uu  : List[SingularParameter]       = generateParamsList(config.outDim, config.outDim, config.maxArity)
  private val bu  : SingularParameter             = addParameters(config.outDim)

  private val Wf  : SingularParameter             = addParameters((config.outDim,config.inDim))
  private val Ufs : List[List[SingularParameter]] = (1 to config.maxArity).toList.map{_ => generateParamsList(config.outDim, config.outDim, config.maxArity)}
  private val bf  : SingularParameter             = addParameters(config.outDim)

  private val initCompressor : Layer              = SingleLayer.compressor(config.inDim, config.outDim)

  private def generateParamsList(inDim:Int, outDim:Int, count:Int) : List[SingularParameter] =
    (1 to count).toList.map{_ => addParameters((inDim, outDim))}

  override def compose(childrenStates: List[State], parentRep: Expression): State = {
    assert(childrenStates.size <= config.maxArity)
    val hs = childrenStates.asInstanceOf[List[TreeLSTMState]].map{_.h}
    val cs = childrenStates.asInstanceOf[List[TreeLSTMState]].map{_.c}
    val i  = superDuoLayer(sigmoid, Wi, parentRep, Ui, hs, bi)
    val o  = superDuoLayer(sigmoid, Wo, parentRep, Uo, hs, bo)
    val fs = Ufs.map(superDuoLayer(sigmoid, Wf, parentRep, _, hs, bf))
    val u  = superDuoLayer(tanh, Wu, parentRep, Uu, hs, bu)
    val c  = i⊙u + (fs zip cs).map{case (f, c) => dropout(f ⊙ c, config.dropout)}.esum // this is old fashioned dropout() from "Recurrent Dropout without Memory Loss"
    val h  = o ⊙ tanh(c)
    TreeLSTMState(h, c)
  }

  override def initState(h: Expression): State =
    TreeLSTMState(initCompressor(h), zeros(config.outDim))

  private case class TreeLSTMState(h:Expression, c:Expression) extends StateClosed

  private def superDuoLayer(a:Activation, W:Expression, x:Expression, Us:List[Expression], hs:List[Expression], b:Expression) : Expression =
    a(W*x + (Us zip hs).map{case (u, h) => u*h}.esum + b)

}
