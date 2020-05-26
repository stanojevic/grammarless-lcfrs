package edin.nn.sequence

import edin.nn.masking.{SentMask, WordMask}
import edin.nn.model.YamlConfig
import edin.nn.DyFunctions._
import edin.nn.embedder.{EmbedderPositionsLearnedConfig, EmbedderPositionsSinusoidConfig}
import edu.cmu.dynet.{Dim, Expression, ParameterCollection}

final case class MultiTransformerConfig(
                                         layers            : Int,
                                         withPositionEmbs  : Boolean,
                                         singleLayerConfig : SequenceEncoderConfig,
                                       ) extends SequenceEncoderConfig {
  override val outDim: Int = singleLayerConfig.outDim
  def construct()(implicit model: ParameterCollection) : SequenceEncoder = new MultiTransformer(this)
}

object MultiTransformerConfig{

  def fromYaml(conf:YamlConfig) : MultiTransformerConfig =
    MultiTransformerConfig(
      layers            = conf("layers").int,
      withPositionEmbs  = conf.getOrElse("with-position-embs", true),
      singleLayerConfig = SingleTransformerConfig.fromYaml(conf)
    )

}

class MultiTransformer(config: MultiTransformerConfig)(implicit model: ParameterCollection) extends SequenceEncoder  {

  override val outDim: Int = config.singleLayerConfig.outDim

  private val layers = (1 to config.layers).toList.map(_ => config.singleLayerConfig.construct())

//  private val positionEmbedder = EmbedderPositionsSinusoidConfig(outDim, 0f).construct()
  private val positionEmbedder = EmbedderPositionsLearnedConfig(100, outDim, 0f).construct()

  override def transduceSentMatrix(sentMatrix: Expression, mask: SentMask): Expression = {
    val in = if(config.withPositionEmbs){
      val n = sentMatrix.cols
      val posEmbs = positionEmbedder.apply((0 until n).toList).reshape(Dim(List(outDim, n)))
      sentMatrix+posEmbs
    }else{
      sentMatrix
    }
    layers.foldLeft(in)((in, layer) => layer.transduceSentMatrix(in, mask))
  }

  override def transduce(xs: List[Expression], masks: List[WordMask]): List[Expression] =
    transduceSentMatrix(concatCols(xs:_*), SentMask(masks)).splitCols

}
