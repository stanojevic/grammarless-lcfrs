package edin.nn.contextualized

import edin.algorithms.AutomaticResourceClosing._
import edin.nn.DyFunctions
import edin.nn.DyFunctions._
import edin.nn.layers.{Dropout, IdentityLayer, SingleLayer}
import edin.nn.masking.WordMask
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

import scala.collection.mutable.{Map => MutMap}

case class ExternalConfig(
                                       dropout                 : Float,
                                       origDim                 : Int,
                                       outDim                  : Int,
                                     ) extends SequenceEmbedderGeneralConfig[String]{
  override def construct()(implicit model: ParameterCollection): SequenceEmbedderGeneral[String] = new External(this)
}

object ExternalConfig{

  def fromYaml(origConf:YamlConfig) : SequenceEmbedderGeneralConfig[String] =
    ExternalConfig(
      dropout            = origConf("dropout"  ).float,
      origDim            = origConf("orig-dim"  ).int,
      outDim             = origConf("out-dim"  ).int
    )

}

object External{

  private val cache = MutMap[List[String], List[Array[Float]]]()

  def clearCache() : Unit =
    cache.clear()

  def addToCache(sent: List[String], embs: List[Array[Float]]) : Unit = {
    require(embs.map(_.length).toSet.size == 1)
    cache(sent) = embs
  }

  def lookup(sent: List[String]) : List[Array[Float]] =
    cache(sent)

  def loadEmbeddings(fn:String) : Unit = {
    var sent : List[String] = Nil
    var vectors = List[Array[Float]]()
    for(line <- linesFromFile(fn)){
      if(line startsWith "sent: "){
        if(vectors != Nil){
          cache(sent) = vectors.reverse
          vectors = Nil
        }
        sent = line.substring(6).trim.split(" +").toList
      }else{
        vectors ::= line.split(" +").map(_.toFloat)
      }
    }
    if(vectors != Nil)
      cache(sent) = vectors.reverse
  }

}

class External(config: ExternalConfig)(implicit model: ParameterCollection) extends SequenceEmbedderGeneral[String] {

  private val compressor = if(config.origDim == config.outDim) IdentityLayer() else SingleLayer.compressor(config.origDim, config.outDim)
  private val drop       = Dropout(config.dropout)

  override def transduceBatchDirect(sents: List[List[String]]): (List[Expression], List[WordMask]) = {
    // old   val embs = sents.map(sent => External.lookup(sent).map(vector))
    // old   val (exps1, masks1) = WordMask.fromSentsExp(embs)
    val embs = sents.map(External.lookup)
    val (exps1, masks1) = WordMask.fromSents(embs)
    val exps2 = exps1.map{ x => drop(compressor(x))}
    (exps2, masks1)
  }

  override def zeros: Expression = DyFunctions.zeros(config.outDim)

}

