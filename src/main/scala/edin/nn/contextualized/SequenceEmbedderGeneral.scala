package edin.nn.contextualized

import edin.nn.DynetSetup
import edin.nn.embedder.{EmbedderCharConcatConfig, EmbedderCharLSTMConfig, EmbedderStandardConfig}
import edin.nn.masking.WordMask
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

import scala.collection.mutable.{Map => MutMap}

/**
  * This is abstract stuff or creating (possibly non-incremental) sequence embedders
  */

object SequenceEmbedderGeneralConfig{

  def fromYaml[K](origConf:YamlConfig) : SequenceEmbedderGeneralConfig[K] =
    origConf("seq-emb-type").str match {
      case "local"        => LocalConfig          .fromYaml(origConf)
      case "global"       => GlobalConfig         .fromYaml(origConf)
      case "combine"      => GeneralCombinerConfig.fromYaml(origConf)
//      case "ELMo"         => ELMoConfig           .fromYaml(origConf).asInstanceOf[SequenceEmbedderGeneralConfig[K]]
      case "External"     => ExternalConfig       .fromYaml(origConf).asInstanceOf[SequenceEmbedderGeneralConfig[K]]
//      case "surface"      => TransformersConfig   .fromYaml(origConf).asInstanceOf[SequenceEmbedderGeneralConfig[K]]
      case "raw" =>
        // this is just to make experimenting easier
        // you also need to add extra fields in yaml w2i: RESOURCE_W2I_TGT_EMBED
        LocalConfig[String]( embConf = EmbedderStandardConfig.fromYaml(origConf) ).asInstanceOf[SequenceEmbedderGeneralConfig[K]]
      case "char" =>
        // this is just to make experimenting easier
        // you also need to add extra fields in yaml w2i: RESOURCE_W2I_TGT_EMBED
        LocalConfig[String]( embConf = EmbedderCharLSTMConfig.fromYaml(origConf) ).asInstanceOf[SequenceEmbedderGeneralConfig[K]]
      case "charconcat" =>
        // this is just to make experimenting easier
        // you also need to add extra fields in yaml w2i: RESOURCE_W2I_TGT_EMBED
        LocalConfig[String]( embConf = EmbedderCharConcatConfig.fromYaml(origConf) ).asInstanceOf[SequenceEmbedderGeneralConfig[K]]
      case "elmo-top" =>
        // this is just to make experimenting easier
        ELMoConfig(
          embeddingType      = "concat_top",
          normalize          = origConf("normalize").bool,
          dropout            = origConf("dropout").float,
          outDim             = origConf("out-dim").int
        ).asInstanceOf[SequenceEmbedderGeneralConfig[K]]
      case "elmo-top-incremental" =>
        // this is just to make experimenting easier
        ELMoConfig(
          embeddingType      = "forward_top",
          normalize          = origConf("normalize").bool,
          dropout            = origConf("dropout").float,
          outDim             = origConf("out-dim").int
        ).asInstanceOf[SequenceEmbedderGeneralConfig[K]]
      case modelName =>
        TransformersConfig(
          modelName          = modelName,
          normalize          = origConf("normalize").bool,
          compressionType    = origConf("compression-type").str,   // "compress-sum" "weighted-sum" "smoothed" "sum-0-1-2-3"
          dropout            = origConf.getOrElse("dropout", 0f),
          outDim             = origConf("out-dim").int
        ).asInstanceOf[SequenceEmbedderGeneralConfig[K]]
    }

}

trait SequenceEmbedderGeneralConfig[T] {

  val outDim : Int

  def construct()(implicit model: ParameterCollection): SequenceEmbedderGeneral[T]

}


trait SequenceEmbedderGeneral[T] {

  // vector of zeros with the same dimension as the output hidden state of the embedder
  def zeros : Expression

  // method that does real embedding of a minibatch ; this is different from embedBatchDirect by Precomputing trait because this includes also compressing of embeddings
  protected def transduceBatchDirect(sents: List[List[T]]) : (List[Expression], List[WordMask])

  final def fakeIncrementalEmbedderBatch(xs: List[List[T]]) : IncrementalEmbedderState[T] = {
    val (exps, masks) = transduceBatch(xs)
    new FakeIncrementalEmbedderState(exps, masks)
  }

  private final var latestCG      = -1
  private final var subCache      = MutMap[List[T], (List[Expression], List[WordMask], Int)]()

  final def transduce(words: List[T]) : List[Expression] = {
    if(latestCG != DynetSetup.cg_id || !subCache.contains(words)){
      transduceBatch(List(words))
    }
    val (exps, masks, i) = subCache(words)
    for{
      (e, m) <- exps zip masks
      wExp   <- m.extractWord(e, i)
    } yield wExp
  }

  final def transduceBatch(sents: List[List[T]]) : (List[Expression], List[WordMask]) = {
    val (batchExp:List[Expression], batchMask:List[WordMask]) = transduceBatchDirect(sents)

    if(latestCG != DynetSetup.cg_id){
      subCache = MutMap()
    }
    latestCG = DynetSetup.cg_id
    for((sent, i) <- sents.zipWithIndex){
      subCache(sent) = (batchExp, batchMask, i)
    }

    (batchExp, batchMask)
  }

}

private class FakeIncrementalEmbedderState[T](embs:List[Expression], masks:List[WordMask]) extends IncrementalEmbedderState[T] {

  override def nextStateAndEmbedBatch(xs: List[Option[T]]): (IncrementalEmbedderState[T], Expression, WordMask) =
    (new FakeIncrementalEmbedderState[T](embs.tail, masks.tail), embs.head, masks.head)
}

