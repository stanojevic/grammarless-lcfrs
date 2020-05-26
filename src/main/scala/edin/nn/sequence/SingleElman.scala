package edin.nn.sequence

import edin.nn.layers.{Dropout, SingleLayerConfig}
import edu.cmu.dynet.{Expression, ParameterCollection}
import edin.nn.DyFunctions._
import edin.nn.masking.WordMask

final case class SingleElmanConfig(
                                   inDim          : Int ,
                                   outDim         : Int ,
                                   dropout        : Float,
                                   dropoutType    : String,
                                   withLayerNorm  : Boolean,
                                 ){
  def construct()(implicit model: ParameterCollection) = new SingleElman(this)
}

final class SingleElman(config:SingleElmanConfig)(implicit model: ParameterCollection) extends RecurrentNN {

  override val outDim: Int = config.outDim

  private val initHH     = addParameters(config.outDim, initAround(0))

  private val useVariationalDropout = config.dropoutType.toLowerCase == "variational"
  private val useDropConnect        = config.dropoutType.toLowerCase == "dropconnect"
  require(useVariationalDropout || useDropConnect, s"unknown type of elman dropout ${config.dropoutType}")

  private val combiner   = SingleLayerConfig(inDim=config.inDim+config.outDim, outDim=config.outDim, activationName="sigmoid", withLayerNorm=config.withLayerNorm, dropout=if(useDropConnect)config.dropout else 0f, dropConnect=useDropConnect).construct()
  private val compressor = SingleLayerConfig(inDim=             config.outDim, outDim=config.outDim, activationName="tanh"   , withLayerNorm=config.withLayerNorm, dropout=config.dropout).construct()
  private val dropInput  = Dropout(dropProb=config.dropout                                  , sameForAllApplications=useVariationalDropout)
  private val dropHidden = Dropout(dropProb=if(useVariationalDropout) config.dropout else 0f, sameForAllApplications=useVariationalDropout)

  override def initState(): RecurrentState =
    new StateElman(initHH)

  final class StateElman(
                         hh    : Expression  , // this is the hidden layer
                       ) extends RecurrentState {

    lazy val h = compressor(hh) // this is the output layer realy

    override val outDim: Int = config.outDim

    override def addInput(i: Expression, mask: WordMask): StateElman =
      new StateElman( mask.fillMask(
          exp    = combiner(concat(dropInput(i), dropHidden(h))),
          filler = initHH) )

  }

}
