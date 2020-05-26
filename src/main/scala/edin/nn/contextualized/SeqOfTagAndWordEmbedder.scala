package edin.nn.contextualized

import edin.nn.masking.WordMask
import edin.nn.DyFunctions._
import edin.nn.model.YamlConfig
import edin.nn.sequence.SequenceEncoderConfig
import edu.cmu.dynet.{Expression, ParameterCollection}


final case class SeqOfTagAndWordEmbedderConfig(
                                                wordEmbedderConfig : SequenceEmbedderGeneralConfig[String],
                                                tagEmbedderConfig  : SequenceEmbedderGeneralConfig[String],
                                                smootherConfig     : SequenceEncoderConfig,
                                              ){
  def construct()(implicit model: ParameterCollection) : SeqOfTagAndWordEmbedder = new SeqOfTagAndWordEmbedder(this)
}

object SeqOfTagAndWordEmbedderConfig{
  def fromYaml(conf:YamlConfig)(implicit model: ParameterCollection) : SeqOfTagAndWordEmbedderConfig =
    SeqOfTagAndWordEmbedderConfig(
      wordEmbedderConfig = SequenceEmbedderGeneralConfig.fromYaml(conf("word-seq-emb")),
      tagEmbedderConfig  = SequenceEmbedderGeneralConfig.fromYaml(conf("tag-seq-emb")),
      smootherConfig     = SequenceEncoderConfig.fromYaml(conf("smoother"))
    )
}

class SeqOfTagAndWordEmbedder(config: SeqOfTagAndWordEmbedderConfig)(implicit model: ParameterCollection){

  private val wordEmbedder = config.wordEmbedderConfig.construct()
  private val tagEmbedder  = if(config.tagEmbedderConfig.outDim<=0) null else config.tagEmbedderConfig.construct()
  private val smoother     = config.smootherConfig.construct()

  def transduce(words:List[String], tags:List[String]) : List[Expression] =
    transduceBatch(List(words), List(tags))._1

  def transduceBatch(batchOfWords:List[List[String]], batchOfTags:List[List[String]]) : (List[Expression], List[WordMask]) = {
    val (we, masks) = wordEmbedder.transduceBatch(batchOfWords)
    val ee = if(tagEmbedder == null){
      we
    }else{
      val te = tagEmbedder.transduceBatch(batchOfTags)._1
      (we zip te).map{case (w, t) => w concat t}
    }
    val ee2 = smoother.transduce(ee)
    (ee2, masks)
  }

}
