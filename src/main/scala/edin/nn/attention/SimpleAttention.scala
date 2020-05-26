package edin.nn.attention

import edu.cmu.dynet.{Expression, ParameterCollection}
import edin.nn.DyFunctions._
import edin.nn.layers.{Layer, MLPConfig}
import edin.nn.masking.SentMask
import edin.nn.model.YamlConfig

sealed case class SimpleAttentionConfig(
                                         attType       : String,
                                         contextDim    : Int,
                                         triggerDim    : Int,
                                         withLayerNorm : Boolean
                                       ){
  def construct()(implicit model:ParameterCollection): SimpleAttention =
    attType.toLowerCase match {
      case "dot"      => new DotAttention(this)
      case "mlp"      => new MLPAttention(this)
      case "bilinear" => new BiLinearAttention(this)
    }
}

object SimpleAttentionConfig{

  def fromYaml(conf:YamlConfig) : SimpleAttentionConfig =
    SimpleAttentionConfig(
      attType       = conf("attention-type").str,
      contextDim    = conf("context-dim").int,
      triggerDim    = conf("trigger-dim").int,
      withLayerNorm = conf.getOrElse("with-layer-norm", false),
    )

}


trait SimpleAttention {

  val config: SimpleAttentionConfig
  lazy val contextDim: Int = config.contextDim
  lazy val triggerDim: Int = config.contextDim

  /**
    * inputMatrix  (D,N)xB
    * targetVector (T)xB
    * return       (N)xB
    */
  def unnormalizedScores(inputMatrix:Expression, targetVector:Expression) : Expression

  /**
    * inputMatrix  (D,N)xB
    * targetVector (T)xB
    * return       (N)xB
    */
  final def normalizedScores(inputMatrix:Expression, targetVector:Expression, mask:SentMask) : Expression =
    softmax(trans(mask.fillMask(trans(unnormalizedScores(inputMatrix, targetVector)), -∞)))

  /**
    * inputMatrix  (D,N)xB
    * targetVector (T)xB
    * return       (D)xB
    */
  final def contextVec(inputMatrix:Expression, targetVector:Expression, mask:SentMask) : Expression =
    inputMatrix*normalizedScores(inputMatrix, targetVector, mask)

  // faster version of transpose for the simple case of vectors
  protected final def trans(v:Expression): Expression =
    v.reshape((v.cols, v.rows), v.batchSize)

}

private class MLPAttention(val config:SimpleAttentionConfig)(implicit model:ParameterCollection) extends SimpleAttention{
  private val mlp:Layer = MLPConfig(
    activations    = List("tanh", "linear") ,
    sizes          = List(config.contextDim+config.triggerDim, config.contextDim, 1),
    withLayerNorm  = config.withLayerNorm   ,
  ).construct
  override def unnormalizedScores(inputMatrix: Expression, targetVector:Expression): Expression = {
    val sentLen = inputMatrix.cols
    val expandedTargetSeq = for(_<- 0 until sentLen) yield targetVector
    val expandedTarget = concatCols(expandedTargetSeq:_*)
    mlp(concat(inputMatrix, expandedTarget)).T
  }
}

private class DotAttention(val config:SimpleAttentionConfig)(implicit model:ParameterCollection) extends SimpleAttention{
  assert(config.contextDim == config.triggerDim, "source and target vector for dot attention must be of the same dimension")
  override def unnormalizedScores(inputMatrix: Expression, targetVector:Expression): Expression =
    trans(targetVector.T*inputMatrix)
}

private class BiLinearAttention(val config:SimpleAttentionConfig)(implicit model:ParameterCollection) extends SimpleAttention{
  private val W = addParameters((config.contextDim, config.triggerDim))
  override def unnormalizedScores(inputMatrix: Expression, targetVector:Expression): Expression =
    inputMatrix.T*W*targetVector
}

