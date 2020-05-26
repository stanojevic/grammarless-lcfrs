package edin.nn.sequence

import edin.nn.DyFunctions._
import edin.nn.layers.{IdentityLayer, SingleLayer}
import edin.nn.masking.WordMask
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

sealed case class MultiBiDirRNNConfig(
                               rnnType        : String,
                               inDim          : Int,
                               outDim         : Int,
                               layers         : Int,
                               withResidual   : Boolean,
                               withLayerNorm  : Boolean,
                               dropProb       : Float,
                               dropType       : String = "variational",
                             ) extends SequenceEncoderConfig{

  require(outDim % 2 == 0)

  def construct()(implicit model: ParameterCollection) : SequenceEncoder = new MultiBiDirRNN(this)
}

object MultiBiDirRNNConfig{

  def fromYaml(conf:YamlConfig) : MultiBiDirRNNConfig =
    MultiBiDirRNNConfig(
      rnnType           = conf("rnn-type").str,
      inDim             = conf("in-dim"  ).int,
      outDim            = conf("out-dim" ).int,
      layers            = conf("layers" ).int,
      withResidual      = conf.getOrElse("with-residual"  , false),
      withLayerNorm     = conf.getOrElse("with-layer-norm", false),
      dropProb          = conf.getOrElse("dropout", 0f),
      dropType          = conf.getOrElse("dropout-type", "variational"),
    )

}

class MultiBiDirRNN(c:MultiBiDirRNNConfig)(implicit model: ParameterCollection) extends SequenceEncoder {

  private val compressor = if(c.withResidual && c.inDim == c.outDim) SingleLayer.compressor(c.inDim, c.outDim) else IdentityLayer()

  override val outDim: Int = c.outDim

  private val rnnPairs = {
    def constructSingle(inDim : Int) :  RecurrentNN               = RecurrentNN.singleFactory(c.rnnType, inDim, c.outDim/2, c.dropProb, c.dropType, c.withLayerNorm)
    def constructPair(  inDim : Int) : (RecurrentNN, RecurrentNN) = ( constructSingle(inDim), constructSingle(inDim) ) // (forwardRnn, backwardRnn)
    constructPair(c.inDim) :: (1 until c.layers).map(_ => constructPair(c.outDim)).toList
  }

  private def applyPair(pair:(RecurrentNN, RecurrentNN), x:List[Expression], mask:List[WordMask]) : List[Expression] = {
    val fwdRep = pair._1.transduce(x, mask)
    val bckRep = pair._2.transduceBackward(x, mask)
    (fwdRep, bckRep).zipped.map{ case (f, b) => concat(f, b) }
  }

  def transduce(xs:List[Expression], mask:List[WordMask]) : List[Expression] =
    rnnPairs.foldLeft(xs map compressor.apply){ (x, pair) => applyPair(pair, x, mask)}

}
