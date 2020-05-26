package edin.nn.contextualized

import edin.nn.DyFunctions
import edin.nn.DyFunctions._
import edin.nn.masking.WordMask
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

case class GeneralCombinerConfig[T](
                                     combiningMethod    : GeneralCombinerConfig.CombiningMethod,
                                     subEmbedderConfigs : List[SequenceEmbedderGeneralConfig[T]],
                                     dropProb           : Float
                                    ) extends SequenceEmbedderGeneralConfig[T]{
  import GeneralCombinerConfig._
  override val outDim: Int = combiningMethod match {
      case Sum | Average =>
        val sizes = subEmbedderConfigs.map(_.outDim)
        require(sizes.forall(_ == sizes.head))
        sizes.head
      case Concat =>
        val sizes = subEmbedderConfigs.map(_.outDim)
        sizes.sum
    }
  override def construct()(implicit model: ParameterCollection): SequenceEmbedderGeneral[T] = new GeneralCombiner[T](this)
}

object GeneralCombinerConfig{

  sealed trait CombiningMethod
  case object Average extends CombiningMethod
  case object Sum     extends CombiningMethod
  case object Concat  extends CombiningMethod

  def fromYaml[K](conf:YamlConfig) : SequenceEmbedderGeneralConfig[K] = {
    val combiningMethod = conf("combining-method").str match {
      case "sum"     => Sum
      case "average" => Average
      case "concat"  => Concat
    }
    val subEmbedders = conf("subembs").list.map(x => SequenceEmbedderGeneralConfig.fromYaml[K](x))
    GeneralCombinerConfig[K](
      combiningMethod    = combiningMethod,
      dropProb           = conf.getOrElse("dropout", 0f),
      subEmbedderConfigs = subEmbedders
    )
  }

}

class GeneralCombiner[T](config : GeneralCombinerConfig[T])(implicit model: ParameterCollection) extends SequenceEmbedderGeneral[T] {

  import GeneralCombinerConfig._

  private val subEmbedders = config.subEmbedderConfigs.map(_.construct())

  override protected def transduceBatchDirect(sents: List[List[T]]): (List[Expression], List[WordMask]) = {
    val subResults  = subEmbedders.map(_.transduceBatch(sents))
    val exps        = subResults.map(_._1).transpose.map(combine)
    val masks       = subResults.head._2
    (exps, masks)
  }

  private def combine(embs:Seq[Expression]) : Expression = config.combiningMethod match {
    case Average => embs.eavg
    case Sum     => embs.esum
    case Concat  => concatSeq(embs)
  }

  override def zeros: Expression = DyFunctions.zeros(config.outDim)

}
