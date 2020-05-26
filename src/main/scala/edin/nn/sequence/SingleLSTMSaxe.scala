package edin.nn.sequence

import edin.general.RichScala._
import edin.nn.DyFunctions._
import edin.nn.layers.{Dropout, IdentityLayer, LayerNorm, SingleLayer}
import edin.nn.masking.WordMask
import edu.cmu.dynet._

final class SingleLSTMSaxe(config:SingleLSTMConfig)(implicit model: ParameterCollection) extends RecurrentNN {

  private  val inDim  : Int = config.inDim
  override val outDim : Int = config.outDim

  private val useVariationalDropout = config.dropoutType.toLowerCase == "variational"
  private val useDropConnect        = config.dropoutType.toLowerCase == "dropconnect"
  private val useMemorySafe         = config.dropoutType.toLowerCase == "memorysafe"
  require(useVariationalDropout || useDropConnect || useMemorySafe, s"unknown type of lstm dropout ${config.dropoutType}")

  private final class DuoLayerSaxe(activationName:String){
    private val norm = if(config.withLayerNorm) LayerNorm(outDim) else IdentityLayer()
    private val activation = activationFactory(activationName)
    private val WA = addParameters((config.outDim, config.outDim), initSaxe(config.outDim, activationName))
    private val WB = addParameters((config.outDim, config.outDim), initSaxe(config.outDim, activationName), dropConnect=if(useDropConnect) config.dropout else 0f)
    private val b  = addParameters(config.outDim)
    def apply(i:Expression, h:Expression) : Expression = activation(norm(WA*i+WB*h+b))
  }

  private val compressor = if(inDim == outDim) IdentityLayer() else SingleLayer.compressor(inDim, outDim)

  private val initH     = addParameters(outDim, initAround(0))
  private val initC     = addParameters(outDim, initAround(0))

  private val iGater = new DuoLayerSaxe("sigmoid")
  private val oGater = new DuoLayerSaxe("sigmoid")
  private val cGater = new DuoLayerSaxe("tanh")
  private val fGater = if(config.coupledForgetGate) null else new DuoLayerSaxe("sigmoid")

  private val dropInput   = Dropout(dropProb=config.dropout                              , sameForAllApplications=useVariationalDropout)
  private val dropHidden  = Dropout(dropProb=(useVariationalDropout? config.dropout | 0f), sameForAllApplications=useVariationalDropout)
  private val dropMemory  = Dropout(dropProb=(useMemorySafe? config.dropout | 0f))

  def initState() : StateLSTM = new StateLSTM(initH, initC)

  final class StateLSTM(
                         val h               : Expression,
                         c                   : Expression,
                       ) extends RecurrentState {

    override val outDim: Int = config.outDim

    override def addInput(i: Expression, mask: WordMask): StateLSTM = {
      val hh = dropHidden(h)
      val ii = dropInput( compressor(i))

      val i_gate  = iGater(ii, hh)
      val o_gate  = oGater(ii, hh)
      val c_prime = cGater(ii, hh)
      val f_gate  = if(config.coupledForgetGate) 1-i_gate else fGater(ii, hh)

      val c_new   = dropMemory(c_prime ⊙ i_gate) + c ⊙ f_gate
      val h_new   = tanh(c_new) ⊙ o_gate
      new StateLSTM(
        h = mask.fillMask(h_new, initH),
        c = mask.fillMask(c_new, initC)
      )
    }

  }

}

