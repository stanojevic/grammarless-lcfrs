package edin.nn.contextualized

import edin.nn.DyFunctions
import edin.nn.DyFunctions._
import edin.nn.masking.WordMask
import edin.nn.model.YamlConfig
import edin.nn.sequence.{MultiUniDirRNNConfig, RecurrentState}
import edu.cmu.dynet.{Expression, ParameterCollection}

/**
  * Directional recurrent embedder
  */
sealed case class DirectionalRecurrentConfig[T](
                                                      subEmbConf:SequenceEmbedderIncrementalConfig[T],
                                                      multRNNconf:MultiUniDirRNNConfig
                                                    ) extends SequenceEmbedderIncrementalConfig[T] {
  override val outDim: Int = multRNNconf.outDim
  override def construct()(implicit model: ParameterCollection): SequenceEmbedderIncremental[T] = new DirectionalRecurrent[T](this)
}

object DirectionalRecurrentConfig{

  def fromYaml[K](conf:YamlConfig) : SequenceEmbedderIncrementalConfig[K] =
    DirectionalRecurrentConfig(
      subEmbConf = SequenceEmbedderIncrementalConfig.fromYaml[K](conf("sub-embedder-conf")),
      multRNNconf = MultiUniDirRNNConfig.fromYaml(conf("recurrent-conf"))
    )

}

class DirectionalRecurrent[T](config: DirectionalRecurrentConfig[T])(implicit model: ParameterCollection) extends SequenceEmbedderIncremental[T] {

  private val embedder = config.subEmbConf.construct()
  private val rnn      = config.multRNNconf.construct()

  override def zeros : Expression = DyFunctions.zeros(config.multRNNconf.outDim)

  override def initState(): IncrementalEmbedderState[T] = new SequenceEmbedderRecurrentState[T](rnn.initState(), embedder.initState(), config.outDim)

}

class SequenceEmbedderRecurrentState[T](rnnState:RecurrentState, subState:IncrementalEmbedderState[T], outDim:Int) extends IncrementalEmbedderState[T] {

  override def nextStateAndEmbedBatch(xs: List[Option[T]]): (IncrementalEmbedderState[T], Expression, WordMask) = {
    val (newSubState, newSubExp, newSubMask) = subState.nextStateAndEmbedBatch(xs)
    val newRnnState = rnnState.addInput(newSubExp, newSubMask)
    val newState = new SequenceEmbedderRecurrentState(newRnnState, newSubState, outDim)
    (newState, newRnnState.h, newSubMask)
  }

}
