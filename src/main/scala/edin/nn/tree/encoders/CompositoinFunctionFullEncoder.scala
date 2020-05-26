package edin.nn.tree.encoders

import edin.nn.tree.EncodableNode
import edin.nn.tree.composers.{CompositionFunction, MultiCompositionFunctionConfig}
import edu.cmu.dynet.ParameterCollection

sealed case class CompositionFunctionFullEncoderConfig(
                                                        compositionType   : String,
                                                        inDim             : Int,
                                                        outDim            : Int,
                                                        maxArity          : Int,
                                                        ignoreHigherInput : Boolean,
                                                        withLayerNorm     : Boolean,
                                                        dropout           : Float
                                                      ){
  def construct()(implicit model: ParameterCollection) = new CompositoinFunctionFullEncoder(this)
}


class CompositoinFunctionFullEncoder(c:CompositionFunctionFullEncoderConfig)(implicit model: ParameterCollection) extends TreeEncoder {

  private val composer = MultiCompositionFunctionConfig(
    compositionType   = c.compositionType,
    inDim             = c.inDim,
    outDim            = c.outDim,
    layers            = 1,
    maxArity          = c.maxArity,
    ignoreHigherInput = c.ignoreHigherInput,
    withLayerNorm     = c.withLayerNorm,
    dropout           = c.dropout
  ).construct()

  override def reencode(root: EncodableNode): Unit =
    recEncode(root)

  private def recEncode(node: EncodableNode): Unit ={
    node.children.foreach(recEncode)
    if(node.isTerm){
      node.nn = composer.initState(node.nn.h)
    }else{
      node.nn = composer.compose(node.children.map{_.nn}, node.nn.h)
    }
  }

}
