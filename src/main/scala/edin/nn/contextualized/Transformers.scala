package edin.nn.contextualized

import edin.general.ipc.Python
import edin.general.{Global => GGlobal}
import edin.nn.DyFunctions
import edu.cmu.dynet.{Expression, ParameterCollection}
import edin.nn.DyFunctions._
import edin.nn.embedder.EmbedderStandardConfig
import edin.nn.layers.{Dropout, MLPConfig, SingleLayer}
import edin.nn.masking.WordMask
import edin.nn.model.YamlConfig
import edin.nn.sequence.MultiBiDirRNNConfig

final case class TransformersConfig(
                               modelName       : String,
                               compressionType : String,
                               normalize       : Boolean,
                               dropout         : Float,
                               outDim          : Int
                             ) extends SequenceEmbedderGeneralConfig[String]{
  override def construct()(implicit model: ParameterCollection): SequenceEmbedderGeneral[String] = new Transformers(this)
}

object TransformersConfig{

  def fromYaml(origConf:YamlConfig) : SequenceEmbedderGeneralConfig[String] =
    origConf("model-name").str.toLowerCase() match {
      case "raw" =>
        // this is just to make experimenting easier
        // you also need to add extra fields in yaml w2i: RESOURCE_W2I_TGT_EMBED
        LocalConfig[String]( embConf = EmbedderStandardConfig.fromYaml(origConf) )
      case "elmo-top" =>
        // this is just to make experimenting easier
        ELMoConfig(
          embeddingType      = "concat_top",
          normalize          = origConf("normalize").bool,
          dropout            = origConf("dropout").float,
          outDim             = origConf("out-dim").int
        )
      case "elmo-top-incremental" =>
        // this is just to make experimenting easier
        ELMoConfig(
          embeddingType      = "forward_top",
          normalize          = origConf("normalize").bool,
          dropout            = origConf("dropout").float,
          outDim             = origConf("out-dim").int
        )
      case _ =>
        TransformersConfig(
          modelName          = origConf("model-name").str,
          normalize          = origConf("normalize").bool,
          compressionType    = origConf("compression-type").str,   // "compress-sum" "weighted-sum" "smoothed" "sum-0-1-2-3"
          dropout            = origConf.getOrElse("dropout", 0f),
          outDim             = origConf("out-dim").int
        )
    }

}

class Transformers(config : TransformersConfig)(implicit model: ParameterCollection) extends SequenceEmbedderGeneral[String] with SequenceEmbedderPrecomputable {

  private def l2_normalize(vec: Array[Float]) : Array[Float] = {
    val l2 = vec.map(x => x*x).sum
    if(l2 == 0)
      vec
    else
      vec.map(_/l2)
  }

  override protected type WordEmbType = List[Array[Float]]
  override def zeros : Expression = DyFunctions.zeros(config.outDim)
  override val name  : String     = config.modelName+"-"+(if(config.normalize) "normalized" else "raw")

  private val vEmbedder = Python.newVarName
  private val vResult   = Python.newVarName
  private val vInput    = Python.newVarName

  private val pythonAccessScript = GGlobal.projectDir+"/scripts/embedding/contextualized.py"

  Python.runPythonFile(pythonAccessScript)
  Python.exec(s"$vEmbedder = constructEmbedder('${config.modelName}')")
  private val transformerDim    = Python.getInt(s"$vEmbedder.dim")
  private val transformerLayers = Python.getInt(s"$vEmbedder.layers")

  private val compressor = CompressorForMultiLayeredEmbedding.compressorFactory(config.compressionType)
  private val drop       = Dropout(config.dropout)

  override def closePrecomputing(): Unit = Python.closeJep()

  protected override def transduceBatchDirect(sents: List[List[String]]) : (List[Expression], List[WordMask]) = {
    // old   val precomputed    : List[List[List[Expression]]]              = findSentsEmb(sents).map(_.map(_.map(vector)).transpose)    // (sentId)(layerId)(wordId)(vector)
    // old   val layersTmp      : List[(List[Expression], List[WordMask])]  = precomputed.transpose.map(WordMask.fromSentsExp)              // (layerId)(sentId)(wordId)(vector)
    val precomputed    : List[List[List[Array[Float]]]]            = findSentsEmb(sents).map(_.transpose)    // (sentId)(layerId)(wordId)(vector)
    val layersTmp      : List[(List[Expression], List[WordMask])]  = precomputed.transpose.map(WordMask.fromSents)         // (layerId)(sentId)(wordId)(vector)
    val masksPerWord   : List[WordMask]                            = layersTmp.head._2                                          // all layers have the same mask  (wordId)(WordMask)
    val layers         : List[List[Expression]]                    = layersTmp.map(_._1)                                        // (layerId)(wordId)(Expression)
    val wordsLayerExps : List[List[Expression]]                    = layers.transpose                                           // (wordId)(layerId)(Expression)
    (compressor(wordsLayerExps, masksPerWord).map(drop(_)), masksPerWord)
  }

  protected def embedBatchDirect(sents: Seq[List[String]]): List[SentEmb] = {
    Python.setList(vInput, sents.map(_ mkString " "))
    Python.exec(s"$vResult = $vEmbedder.embed_batch($vInput)")

    val result = for{ (sent, sentId) <- sents.zipWithIndex.toList } yield
                   for{ i <- sent.indices.toList                      } yield
                     for{ layerId <- (0 until transformerLayers).toList } yield {
                       val vec = Python.getNumPyArray(s"$vResult[$sentId][$layerId][$i]")
                       if(config.normalize)
                         l2_normalize(vec)
                       else
                         vec
                     }

    Python.delVar(vInput)
    Python.delVar(vResult)

    result
  }

  private trait CompressorForMultiLayeredEmbedding{
    // (wordId)(layerId)(Expression)
    def apply(es: List[List[Expression]], masks:List[WordMask]) : List[Expression]
  }

  private object CompressorForMultiLayeredEmbedding{

    def compressorFactory : String => CompressorForMultiLayeredEmbedding = {
      case "compress-sum"           => new CompressorCompressSum
      case "weighted-sum"           => new CompressorWeightedSum
      case "smoothed"               => new CompressorSmoothed
      case x if x startsWith "sum-" => new CompressorSum(x.drop(4).split("-").toList.map(_.toInt))
    }

    private class CompressorCompressSum extends CompressorForMultiLayeredEmbedding{
      private val compressors =
        (0 until transformerLayers).map(_ => SingleLayer.compressor(transformerDim, config.outDim))
      override def apply(es: List[List[Expression]], masks: List[WordMask]): List[Expression] =
        for{
          (layers, mask) <- es zip masks
          res = (layers zip compressors).map{case (layer, compressor) => compressor(layer)}.esum
        } yield res
    }

    private class CompressorWeightedSum extends CompressorForMultiLayeredEmbedding{
      private val weights = addParameters(transformerLayers)
      private val compressor = SingleLayer.compressor(transformerDim, config.outDim)
      override def apply(es: List[List[Expression]], masks: List[WordMask]): List[Expression] =
        for( (layers, mask) <- es zip masks )
          yield compressor(concatCols(layers:_*) * weights.exp)
    }

    private class CompressorSum(layersToSelect:List[Int]) extends CompressorForMultiLayeredEmbedding{
      private val compressor = SingleLayer.compressor(transformerDim, config.outDim)
      override def apply(es: List[List[Expression]], masks: List[WordMask]): List[Expression] =
        for{
          (layers, mask) <- es zip masks
          layersRev = layers.reverse // layersToSelect are numbered top-down (0 is the last layer), but layers is bottom-up
        }
          yield compressor(layersToSelect.map(layerId => layersRev(layerId)).esum)
    }

    private class CompressorSmoothed extends CompressorForMultiLayeredEmbedding {
      private val compressor    = SingleLayer.compressor(transformerDim, config.outDim)
      private val precompressor = new CompressorWeightedSum
      private val smoother      = MultiBiDirRNNConfig(
        rnnType           = "LSTM",
        inDim             = config.outDim,
        outDim            = config.outDim,
        layers            = 1,
        withResidual      = false,
        withLayerNorm     = false,
        dropProb          = 0f
      ).construct()
      private val mlp           = MLPConfig(
        activations = List("relu", "softmax"),
        sizes       = List(config.outDim, config.outDim, transformerLayers),
      ).construct()
      override def apply(es: List[List[Expression]], masks: List[WordMask]): List[Expression] = {
        val attentions = smoother.transduce(precompressor(es, masks), masks).map(mlp(_))
        for(((layers, attention), mask) <- es zip attentions zip masks)
          yield compressor(concatCols(layers:_*) * attention)
      }
    }
  }


}

