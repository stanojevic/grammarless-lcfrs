package edin.nn.contextualized

import edin.nn.DyFunctions
import edin.nn.DyFunctions._
import edin.nn.masking.WordMask
import edin.nn.model.YamlConfig
import edin.nn.sequence._
import edu.cmu.dynet.{Expression, ParameterCollection}

/**
  * This is for an embedder that embeds each word by taking a more global context into account
  * It can be unidirectional lstm, bidirectional lstm, transformer (not implemented yet) etc.
  */

sealed case class GlobalConfig[T](
                                                   subEmbConf : SequenceEmbedderGeneralConfig[T],
                                                   rnnConfig  : SequenceEncoderConfig
                                                 ) extends SequenceEmbedderGeneralConfig[T] {
  override val outDim: Int = rnnConfig.outDim
  override def construct()(implicit model: ParameterCollection): SequenceEmbedderGeneral[T] = new Global[T](this)
}

object GlobalConfig{

  def fromYaml[K](conf:YamlConfig) : SequenceEmbedderGeneralConfig[K] =
    if(conf("recurrent-conf")("layers").int == 0){
      // skip this embedder and expose only sub-embedder
      SequenceEmbedderGeneralConfig.fromYaml[K](
        conf("sub-embedder-conf"))
    }else{
      val rnnYaml   = conf("recurrent-conf")
      val rnnConfig = rnnYaml("bi-directional").bool match {
        case true  => MultiBiDirRNNConfig.fromYaml(rnnYaml)
        case false => MultiUniDirRNNConfig.fromYaml(rnnYaml)
      }
      GlobalConfig(
        subEmbConf  = SequenceEmbedderGeneralConfig.fromYaml[K](conf("sub-embedder-conf")),
        rnnConfig = rnnConfig )
    }

}

class Global[T](config: GlobalConfig[T])(implicit model: ParameterCollection) extends SequenceEmbedderGeneral[T] {

  var subEmbedder : SequenceEmbedderGeneral[T] = config.subEmbConf.construct()
  var rnn : SequenceEncoder = config.rnnConfig.construct()

  override def zeros : Expression = DyFunctions.zeros(config.rnnConfig.outDim)

  protected override def transduceBatchDirect(sents: List[List[T]]) : (List[Expression], List[WordMask]) = {
    val (exps, masks) = subEmbedder.transduceBatch(sents)
    val res = rnn.transduce(exps, masks)
    (res, masks)
  }

}

