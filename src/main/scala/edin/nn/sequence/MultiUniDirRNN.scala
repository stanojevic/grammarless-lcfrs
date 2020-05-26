package edin.nn.sequence

import edin.nn.masking.WordMask
import edin.nn.model.YamlConfig
import edin.nn.DyFunctions._
import edin.nn.layers.{IdentityLayer, SingleLayer}
import edu.cmu.dynet.{Expression, ParameterCollection}

/**
  * Constructs multi-layer recurrent neural net with the same directionality (not bi-directional)
  */
sealed case class MultiUniDirRNNConfig(
                                  rnnType           : String,
                                  inDim             : Int,
                                  outDim            : Int,
                                  layers            : Int,
                                  withResidual      : Boolean,
                                  withLayerNorm     : Boolean,
                                  dropProb          : Float,
                                  dropType          : String="variational",
                                ) extends SequenceEncoderConfig {
  def construct()(implicit model: ParameterCollection) = new MultiUniDirRNN(this)
}

object MultiUniDirRNNConfig{

  def fromYaml(conf:YamlConfig) : MultiUniDirRNNConfig =
    MultiUniDirRNNConfig(
      rnnType           = conf("rnn-type").str,
      inDim             = conf("in-dim"  ).int,
      outDim            = conf("out-dim" ).int,
      layers            = conf("layers"  ).int,
      withResidual      = conf.getOrElse("with-residual"   , false        ),
      withLayerNorm     = conf.getOrElse("with-layer-norm" , false        ),
      dropProb          = conf.getOrElse("dropout"         , 0f           ),
      dropType          = conf.getOrElse("dropout-type"    , "variational")
    )

}



class MultiUniDirRNN(c:MultiUniDirRNNConfig)(implicit model: ParameterCollection) extends RecurrentNN {
  require(c.layers > 0)

  override val outDim: Int = c.outDim
  private val compressor = if(c.withResidual) SingleLayer.compressor(c.inDim, c.outDim) else IdentityLayer()

  private val rnns: List[RecurrentNN] = {
    def createRNN(inDim:Int) : RecurrentNN = RecurrentNN.singleFactory(c.rnnType, inDim, c.outDim, c.dropProb, c.dropType, c.withLayerNorm)
    createRNN(c.inDim) :: (2 to c.layers).map(_ => createRNN(c.outDim)).toList
  }

  override def initState(): RecurrentState = {
    var newStates = List[RecurrentState]()
    newStates ::= rnns.head.initState()
    for(rnn <- rnns.tail){
      newStates ::= rnn.initState().addInput(newStates.head.h)
    }
    new MultiRNNState(newStates.reverse)
  }

  private class MultiRNNState(rnnStates:List[RecurrentState]) extends RecurrentState {

    override val outDim: Int = c.outDim

    override val h = if(c.withResidual && rnnStates.size>1)
      rnnStates.reverse.take(2).map(_.h).esum
    else
      rnnStates.last.h

    override def addInput(x: Expression, m:WordMask): RecurrentState = {
      var newStates = List[RecurrentState]()
      var currOut   = compressor(x)
      for((rnnState, i) <- rnnStates.zipWithIndex){
        newStates ::= rnnState.addInput(currOut, m)
        if(c.withResidual)
          currOut += newStates.head.h
        else
          currOut  = newStates.head.h
      }
      new MultiRNNState(newStates.reverse)
    }

  }

}


