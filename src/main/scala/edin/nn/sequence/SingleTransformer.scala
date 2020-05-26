package edin.nn.sequence

import edin.nn.DyFunctions._
import edin.nn.attention.MultiHeadAttentionConfig
import edin.nn.layers.{LayerNormConfig, MLPConfig}
import edin.nn.masking.{SentMask, WordMask}
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

// TODO Saxe initialization?
final case class SingleTransformerConfig(
                                          modelDim  : Int ,
                                          heads     : Int ,
                                          dropout   : Float,
                                        ) extends SequenceEncoderConfig {
  override val outDim: Int = modelDim
  def construct()(implicit model: ParameterCollection) : SequenceEncoder = new SingleTransformer(this)
}

object SingleTransformerConfig{

  def fromYaml(conf:YamlConfig) : SingleTransformerConfig =
    SingleTransformerConfig(
      modelDim = conf("out-dim").int,
      heads = conf("heads").int,
      dropout = conf.getOrElse("dropout", 0f)
    )

}

class SingleTransformer(config:SingleTransformerConfig)(implicit model: ParameterCollection) extends SequenceEncoder {

  private val multiHeadAttention = MultiHeadAttentionConfig(config.modelDim, config.heads, config.dropout).construct()
  private val ff  = MLPConfig(activations=List("relu", "linear"), sizes=List(config.modelDim, config.modelDim, config.modelDim)).construct()
  private val ln1 = LayerNormConfig(config.modelDim).construct()
  private val ln2 = LayerNormConfig(config.modelDim).construct()

  override val outDim: Int = config.modelDim

  private def selfAttention(sentMatrix:Expression, mask:SentMask) : Expression =
    multiHeadAttention(sentMatrix, sentMatrix, sentMatrix, mask)

  // TODO TO RE-READ and ADD "Improving Deep Transformerwith Depth-Scaled Initialization and Merged Attention" https://arxiv.org/pdf/1908.11365.pdf
  // TODO TO READ            "Tensor2Tensor for Neural Machine Translation" https://www.aclweb.org/anthology/W18-1819.pdf

  // original (but suboptimal) "post-norm" layer normalization
  def transduceSentMatrix_original(sentMatrix: Expression, mask:SentMask) : Expression = {
    val a1 = ln1(sentMatrix+selfAttention(sentMatrix, mask))
    val a2 = ln2(a1+ff(a1))
    a2
  }

  // improved "pre-norm" layer normalization from "Learning Deep Transformer Models for Machine Translation" https://arxiv.org/pdf/1906.01787.pdf
  // and Tensor2Tensor ?
  override def transduceSentMatrix(sentMatrix: Expression, mask:SentMask) : Expression =
    sentMatrix+ff(selfAttention(ln1(sentMatrix), mask))

  override def transduce(xs: List[Expression], masks: List[WordMask]): List[Expression] =
    transduceSentMatrix(concatCols(xs:_*), SentMask(masks)).splitCols

}

