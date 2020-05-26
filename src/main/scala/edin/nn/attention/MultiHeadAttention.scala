package edin.nn.attention

import edin.nn.layers.{Dropout, SingleLayer}
import edin.nn.masking.SentMask
import edin.nn.DyFunctions._
import edu.cmu.dynet.{Expression, ParameterCollection}

final case class MultiHeadAttentionConfig(
                                           modelDim : Int,
                                           heads    : Int,
                                           dropout  : Float=0f
                                         )(implicit model: ParameterCollection){
  def construct()(implicit model: ParameterCollection) = new MultiHeadAttention(this)
}

class MultiHeadAttention(config:MultiHeadAttentionConfig)(implicit model: ParameterCollection){

  private val modelDim  = config.modelDim
  private val headNum   = config.heads
  private val headDim   = modelDim/headNum
  private val kLayer    = SingleLayer.compressor(modelDim, modelDim)
  private val vLayer    = SingleLayer.compressor(modelDim, modelDim)
  private val qLayer    = SingleLayer.compressor(modelDim, modelDim)
  private val oLayer    = SingleLayer.compressor(modelDim, modelDim)
  private val drop      = Dropout(config.dropout)
  assert(config.modelDim%config.heads==0, s"model size ($modelDim) must be divisible by the number of heads ($headNum)")

  def apply(key:Expression, value:Expression, query:Expression, mask:SentMask) : Expression = {
    // k,v,q  ----> (D,N)xB

    val batchSize = key.batchSize
    val sentLen   = key.cols

    val k = kLayer(key  ).T.reshape((sentLen, headDim), headNum*batchSize) // (N, HD)x(HC, B)
    val v = vLayer(value).T.reshape((sentLen, headDim), headNum*batchSize) // (N, HD)x(HC, B)
    val q = qLayer(query).T.reshape((sentLen, headDim), headNum*batchSize) // (N, HD)x(HC, B)

    val scores    = (q*k.T)/math.sqrt(headDim).toFloat // (N, N)x(HC, B)
    val scores2   = mask.fillMask(scores, -∞, headNum)
    val attention = drop(softmax(scores2, 1)) // (N, N)x(HC, B)

    val context = (attention.T*v).reshape((sentLen, modelDim), batchSize) // (N, D)xB

    oLayer(context.T) // (D, N)xB
  }

}

