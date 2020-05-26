package edin.nn.sequence

import edin.general.RichScala._
import edin.nn.DyFunctions._
import edin.nn.layers.{Dropout, IdentityLayer, LayerNorm, SingleLayer}
import edin.nn.masking.WordMask
import edu.cmu.dynet._

final case class SingleLSTMConfig(
                                  inDim             : Int ,
                                  outDim            : Int ,
                                  dropout           : Float   = 0f,
                                  dropoutType       : String  = "variational", // "variational", "dropConnect", "memorySafe"
                                  withLayerNorm     : Boolean = false,
                                  coupledForgetGate : Boolean
                                 ){
  def construct()(implicit model: ParameterCollection) = new SingleLSTM(this)
}

final class SingleLSTM(config:SingleLSTMConfig)(implicit model: ParameterCollection) extends RecurrentNN {

  private  val inDim  : Int = config.inDim
  override val outDim : Int = config.outDim

  private val useVariationalDropout = config.dropoutType.toLowerCase == "variational"
  private val useDropConnect        = config.dropoutType.toLowerCase == "dropconnect"
  private val useMemorySafe         = config.dropoutType.toLowerCase == "memorysafe"
  require(useVariationalDropout || useDropConnect || useMemorySafe, s"unknown type of lstm dropout ${config.dropoutType}")

  private val initH     = addParameters(outDim, initAround(0))
  private val initC     = addParameters(outDim, initAround(0))
  private val big_layer = SingleLayer.compressor(
                                                  inDim=inDim+outDim,
                                                  outDim=(config.coupledForgetGate? (3*outDim) | (4*outDim)),
                                                  dropConnect=useDropConnect
                                                )

  private val dropInput   = Dropout(dropProb=config.dropout                              , sameForAllApplications=useVariationalDropout)
  private val dropHidden  = Dropout(dropProb=(useVariationalDropout? config.dropout | 0f), sameForAllApplications=useVariationalDropout)
  private val dropMemory  = Dropout(dropProb=(useMemorySafe? config.dropout | 0f))

  private val iNorm = if(config.withLayerNorm) LayerNorm(outDim) else IdentityLayer()
  private val oNorm = if(config.withLayerNorm) LayerNorm(outDim) else IdentityLayer()
  private val cNorm = if(config.withLayerNorm) LayerNorm(outDim) else IdentityLayer()
  private val fNorm = if(config.withLayerNorm) LayerNorm(outDim) else IdentityLayer()

  def initState() : StateLSTM = new StateLSTM(initH, initC)

  final class StateLSTM(
                         val h               : Expression,
                         c                   : Expression,
                       ) extends RecurrentState {

    override val outDim: Int = config.outDim

    override def addInput(i: Expression, mask: WordMask): StateLSTM = {
      val hh = dropHidden(h)
      val ii = dropInput( i)

      val big_result = big_layer(concat(ii, hh)).reshape((if(config.coupledForgetGate) 3 else 4, outDim))

      val i_gate  = sigmoid(iNorm(big_result(0)))
      val o_gate  = sigmoid(oNorm(big_result(1)))
      val c_prime =    tanh(cNorm(big_result(2)))
      val f_gate  = if(config.coupledForgetGate) 1-i_gate else sigmoid(fNorm(big_result(3)))

      val c_new   = dropMemory(c_prime ⊙ i_gate) + c ⊙ f_gate
      val h_new   = tanh(c_new) ⊙ o_gate
      new StateLSTM(
        h = mask.fillMask(h_new, initH),
        c = mask.fillMask(c_new, initC)
      )
    }

  }

}

