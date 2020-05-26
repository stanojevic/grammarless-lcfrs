package edin.nn.contextualized

import edin.nn.masking.WordMask
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

trait SequenceEmbedderIncrementalConfig[T] extends SequenceEmbedderGeneralConfig[T] {

  def construct()(implicit model: ParameterCollection): SequenceEmbedderIncremental[T]

}

object SequenceEmbedderIncrementalConfig{

  def fromYaml[K](origConf:YamlConfig) : SequenceEmbedderIncrementalConfig[K] =
    origConf("seq-emb-type").str match {
      case "local"     => LocalConfig.fromYaml(origConf)
      case "recurrent"    => DirectionalRecurrentConfig.fromYaml(origConf)
    }

}

trait SequenceEmbedderIncremental[T] extends SequenceEmbedderGeneral[T] {

  def initState() : IncrementalEmbedderState[T]

  private def makeAllListsEqual(sents:List[List[T]]) : List[List[Option[T]]] = {
    val lengths = sents.map(_.size)
    val maxLen = lengths.max
    for((sent, len) <- sents zip lengths)
      yield sent.map(Some(_)) ++ List.fill(maxLen-len)(None)
  }

  protected override def transduceBatchDirect(sents: List[List[T]]) : (List[Expression], List[WordMask]) = {
    var state = initState()
    var res   = List[Expression]()
    var masks = List[WordMask]()
    for(wordVertical <- makeAllListsEqual(sents).transpose){
      val y = state.nextStateAndEmbedBatch(wordVertical)
      state   = y._1
      res   ::= y._2
      masks ::= y._3
    }
    (res.reverse, masks.reverse)
  }

}

trait IncrementalEmbedderState[T]{

  final def nextStateAndEmbed(x: T): (IncrementalEmbedderState[T], Expression) = {
    val (state, exp, mask) = nextStateAndEmbedBatch(List(Some(x)))
    (state, exp)
  }

  def nextStateAndEmbedBatch(xs: List[Option[T]]): (IncrementalEmbedderState[T], Expression, WordMask)

}

