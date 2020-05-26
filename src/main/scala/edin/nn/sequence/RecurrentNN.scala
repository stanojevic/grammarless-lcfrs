package edin.nn.sequence

import edin.nn.StateClosed
import edin.nn.masking.WordMask
import edu.cmu.dynet.{Expression, ParameterCollection}

trait RecurrentNN extends SequenceEncoder {

  def initState() : RecurrentState

  // keeps the order but only computes representation backward
  final def transduceBackward(vectors:List[Expression]) : List[Expression] =
    transduce(vectors.reverse).reverse

  final def transduce(vectors:List[Expression], mask: List[WordMask]) : List[Expression] =
    this.initState().transduce(vectors, mask)

  // keeps the order but only computes representation backward
  final def transduceBackward(vectors:List[Expression], mask: List[WordMask]) : List[Expression] =
    transduce(vectors.reverse, mask.reverse).reverse
}

object RecurrentNN{

  def singleFactory(
                     rnnType        : String   ,
                     inDim          : Int      ,
                     outDim         : Int      ,
                     dropProb       : Float    ,
                     dropType       : String   , // "variational", "dropConnect", "memorySafe"
                     withLayerNorm  : Boolean  ,
                   )(implicit model:ParameterCollection) : RecurrentNN =
    rnnType.toLowerCase match {
      case "lstm-vanilla" | "lstm" => SingleLSTMConfig(
        inDim             = inDim        ,
        outDim            = outDim       ,
        dropout           = dropProb     ,
        dropoutType       = dropType     ,
        withLayerNorm     = withLayerNorm,
        coupledForgetGate = false
      ).construct
      case "lstm-coupled" => SingleLSTMConfig(
        inDim             = inDim        ,
        outDim            = outDim       ,
        dropout           = dropProb     ,
        dropoutType       = dropType     ,
        withLayerNorm     = withLayerNorm,
        coupledForgetGate = true
      ).construct
      case "lstm-vanilla-saxe" => new SingleLSTMSaxe( SingleLSTMConfig(
        inDim             = inDim        ,
        outDim            = outDim       ,
        dropout           = dropProb     ,
        dropoutType       = dropType     ,
        withLayerNorm     = withLayerNorm,
        coupledForgetGate = false
      ))
      case "lstm-coupled-saxe" => new SingleLSTMSaxe( SingleLSTMConfig(
        inDim             = inDim        ,
        outDim            = outDim       ,
        dropout           = dropProb     ,
        dropoutType       = dropType     ,
        withLayerNorm     = withLayerNorm,
        coupledForgetGate = true
      ))
      case "gru" => SingleGRUConfig(
        inDim          = inDim        ,
        outDim         = outDim       ,
        dropout        = dropProb     ,
        dropoutType    = dropType     ,
        withLayerNorm  = withLayerNorm,
      ).construct
      case "elman" => SingleElmanConfig(
        inDim          = inDim        ,
        outDim         = outDim       ,
        dropout        = dropProb     ,
        dropoutType    = dropType     ,
        withLayerNorm  = withLayerNorm,
      ).construct
    }

}

trait RecurrentState extends StateClosed {

  val outDim : Int

  def addInput(i:Expression, mask:WordMask) : RecurrentState

  final def addInput(i: Expression): RecurrentState =
    addInput(i, WordMask.totallyUnmasked)

  final def transduce(vectors:List[Expression]) : List[Expression] = {
    var output = List[Expression]()
    var currState = this
    for(vec <- vectors){
      currState = currState.addInput(vec)
      output ::= currState.h
    }
    output.reverse
  }

  final def transduce(vectors:List[Expression], masks: List[WordMask]) : List[Expression] = {
    var output = List[Expression]()
    var currState = this
    for((vec, m) <- (vectors zip masks)){
      currState = currState.addInput(vec, m)
      output ::= currState.h
    }
    output.reverse
  }

}

