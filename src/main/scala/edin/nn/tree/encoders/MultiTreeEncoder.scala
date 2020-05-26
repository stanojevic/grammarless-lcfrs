package edin.nn.tree.encoders

import edin.nn.model.Any2Int
import edin.nn.tree._
import edu.cmu.dynet.ParameterCollection

sealed case class MultiTreeEncoderConfig(
                                        layers             : Int             ,
                                        inDim              : Int             ,
                                        outDim             : Int             ,
                                        encoderType        : String          ,
                                        maxArity           : Int             ,
                                        dropout            : Float           ,
                                        withLayerNorm      : Boolean         ,
                                        compositionName    : String          , // if encoderType == "Composer"
                                        ignoreHigherInput  : Boolean         , // if encoderType == "Composer" and uses SpanRep
                                        gcnActivationName  : String          , // if encoderType == "GCN"
                                        gcnGated           : Boolean         , // if encoderType == "GCN"
                                        gcnEdgeSpecificBias: Boolean         , // if encoderType == "GCN"
                                        gcnE2I             : Any2Int[String]   // if encoderType == "GCN"
                                        ){
  def construct()(implicit model: ParameterCollection) = new MultiTreeEncoder(this)
}


class MultiTreeEncoder(c:MultiTreeEncoderConfig)(implicit model: ParameterCollection) extends TreeEncoder {

  private val layers: List[TreeEncoder] = {
    val first = constructTreeLayer(c, initial = true)
    val rest  = (2 to c.layers).map(_ => constructTreeLayer(c, initial = false)).toList
    first :: rest
  }

  override def reencode(root: EncodableNode): Unit =
    layers.foreach(_.reencode(root))

  private def constructTreeLayer(c:MultiTreeEncoderConfig, initial:Boolean)(implicit model: ParameterCollection) : TreeEncoder = {
    val inDim = if(initial) c.inDim else c.outDim
    c.encoderType match {
      case "IORNN"    => SingleIORNNConfig(
                           inDim      = inDim,
                           outDim     = c.outDim,
                           seqType    = "LSTM",
                           withLayerNorm = false,// TODO
                           seqDropout = c.dropout
                         ).construct()
      case "Composer" => CompositionFunctionFullEncoderConfig(
                           compositionType   = c.compositionName,
                           inDim             = inDim,
                           outDim            = c.outDim,
                           maxArity          = c.maxArity,
                           ignoreHigherInput = c.ignoreHigherInput,
                           withLayerNorm     = c.withLayerNorm,
                           dropout           = c.dropout
                         ).construct()
    }
  }

}
