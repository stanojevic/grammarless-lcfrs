package edin.nn.layers

import edu.cmu.dynet.{Expression, ParameterCollection}
import edin.nn.DyFunctions._
import edin.nn.model.YamlConfig

sealed case class CombinerConfig(
                                  inDims           : List[Int],
                                  midDim           : Int,
                                  outDim           : Int,
                                  combType         : String,
                                  activationName   : String,
                                  dropout          : Float,
                                  withPostencoding : Boolean,
                                  withLayerNorm    : Boolean,
                                ){
  def construct()(implicit model: ParameterCollection) = new Combiner(this)
}

object CombinerConfig{

  def fromYaml(conf:YamlConfig) : CombinerConfig = {
    CombinerConfig(
      inDims           = conf("in-dims"        ).intList,
      midDim           = conf("mid-dim"        ).int,
      outDim           = conf("out-dim"        ).int,
      combType         = conf("comb-type"      ).str,
      activationName   = conf("activation"     ).str,
      dropout          = conf.getOrElse("dropout", 0f),
      withPostencoding = conf("with-postencoding").bool,
      withLayerNorm    = conf.getOrElse("with-layer-norm" , false),
    )
  }

}

class Combiner(config:CombinerConfig)(implicit model: ParameterCollection) {

  val outDim : Int = config.outDim

  val finalCompressor : Layer = if(config.withPostencoding){
    SingleLayerConfig(
      inDim = config.midDim,
      outDim = config.outDim,
      activationName = config.activationName,
      withLayerNorm = config.withLayerNorm,
      dropout = config.dropout
    ).construct()
  }else{
    assert(config.midDim == config.outDim)
    IdentityLayer()
  }

  val compressors : List[Layer] = config.combType match {
    case "concat" =>
      assert(config.inDims.sum == config.midDim)
      Nil
    case "sum" =>
      config.inDims.filter(_>0).map( x =>
        SingleLayerConfig(
          inDim = x,
          outDim = config.midDim,
          activationName = config.activationName,
          withLayerNorm = config.withLayerNorm,
          dropout = config.dropout
        ).construct()
      )
  }

  def apply(exps: Expression*): Expression = {
    assert(exps.size == config.inDims.size)
    val preencoding = config.combType match {
      case "concat" =>
        concatSeqWithNull(exps)
      case "sum" =>
        (exps.filter(_!=null) zip compressors).map{case (e, c) => c(e)}.esum
    }
    finalCompressor(preencoding)
  }

}

