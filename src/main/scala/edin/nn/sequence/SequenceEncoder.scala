package edin.nn.sequence

import edin.nn.masking.{SentMask, WordMask}
import edin.nn.DyFunctions._
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

trait SequenceEncoderConfig{

  val outDim:Int

  def construct()(implicit model: ParameterCollection) : SequenceEncoder
}

object SequenceEncoderConfig{

  def fromYaml(conf:YamlConfig)(implicit model: ParameterCollection) : SequenceEncoderConfig =
    (conf("rnn-type").str.toLowerCase, conf("bi-directional").bool) match {
      case ("transformer", true ) => MultiTransformerConfig.fromYaml(conf)
      case (_            , true ) => MultiBiDirRNNConfig.fromYaml(conf)
      case (_            , false) => MultiUniDirRNNConfig.fromYaml(conf)
    }

}

trait SequenceEncoder{

  val outDim:Int

  final def transduceSentMatrix(sentMatrix: Expression) : Expression =
    transduceSentMatrix(sentMatrix, SentMask((0 until sentMatrix.cols).toList.map(_=> WordMask.totallyUnmasked)))

  def transduceSentMatrix(sentMatrix: Expression, mask:SentMask) : Expression =
    concatCols(transduce(sentMatrix.splitCols, mask.wordMasks):_*) // inefficient

  final def transduce(xs:List[Expression]) : List[Expression] =
    transduce(xs, xs.map(_ => WordMask.totallyUnmasked))

  def transduce(xs:List[Expression], mask:List[WordMask]) : List[Expression]

}
