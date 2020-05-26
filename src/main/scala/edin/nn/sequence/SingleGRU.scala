package edin.nn.sequence

import edin.nn.masking.WordMask
import edin.nn.DyFunctions._
import edin.nn.layers.{Dropout, SingleLayerConfig}
import edu.cmu.dynet.{Expression, ParameterCollection}

final case class SingleGRUConfig(
                                   inDim          : Int ,
                                   outDim         : Int ,
                                   dropout        : Float,
                                   dropoutType    : String,
                                   withLayerNorm  : Boolean,
                                 ){
  def construct()(implicit model: ParameterCollection) = new SingleGRU(this)
}

final class SingleGRU(config:SingleGRUConfig)(implicit model: ParameterCollection) extends RecurrentNN {

  sys.error("this code may contain a bug -- check it before use")

  private  val inDim       = config.inDim
  override val outDim: Int = config.outDim

  private val initH       = addParameters(outDim, initAround(0))

  private val useVariationalDropout = config.dropoutType.toLowerCase == "variational"
  private val useDropConnect        = config.dropoutType.toLowerCase == "dropconnect"
  require(useVariationalDropout || useDropConnect, s"unknown type of elman dropout ${config.dropoutType}")

  private val big_layer = SingleLayerConfig(
    inDim          = inDim+outDim,
    outDim         = outDim+outDim,
    activationName = "logistic",
    withLayerNorm  = config.withLayerNorm,
    dropout        = if(useDropConnect) config.dropout else 0f,
    dropConnect    = useDropConnect
  ).construct()

  private val c_gater   = SingleLayerConfig(
    inDim          = inDim+outDim,
    outDim         = outDim,
    activationName = "tanh",
    withLayerNorm  = config.withLayerNorm,
    dropout        = if(useDropConnect) config.dropout else 0f,
    dropConnect    = useDropConnect
  ).construct()

  private val dropInput   = Dropout(dropProb=config.dropout                                  , sameForAllApplications=useVariationalDropout)
  private val dropHidden  = Dropout(dropProb=if(useVariationalDropout) config.dropout else 0f, sameForAllApplications=useVariationalDropout)

  override def initState(): RecurrentState = new StateGRU(initH)

  final class StateGRU(
                         val h : Expression
                       ) extends RecurrentState {

    override val outDim: Int = config.outDim

    override def addInput(i: Expression, mask: WordMask): StateGRU = {
      val hh      = dropHidden(h)
      val ii      = dropInput(i)

      val big_result = big_layer(concat(ii, hh)).reshape((2, outDim))
      val (r, z) = (big_result(0), big_result(1))

      val h_prime = c_gater(concat(ii, r⊙hh))
      val h_new   = z⊙hh+(1-z)⊙h_prime
      new StateGRU(
        h = mask.fillMask(h_new, initH)
      )
    }

  }
}
