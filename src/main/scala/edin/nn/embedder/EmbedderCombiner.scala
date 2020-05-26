package edin.nn.embedder

import edin.nn.layers.{Dropout, Layer, SingleLayer}
import edin.nn.DyFunctions._
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}

case class EmbedderCombinerConfig[T](
                                    outDim             : Int,
                                    subEmbedderConfigs : List[EmbedderConfig[T]],
                                    dropProb           : Float
                                    ) extends EmbedderConfig[T] {
  override def construct()(implicit model: ParameterCollection): Embedder[T] = new EmbedderCombiner[T](this)
}

object EmbedderCombinerConfig{

  def fromYaml[K](conf:YamlConfig) : EmbedderConfig[K] =
    EmbedderCombinerConfig[K](
      outDim = conf("out-dim").int,
      dropProb = conf.getOrElse("dropout", 0f),
      subEmbedderConfigs = conf("subembs").list.map(EmbedderConfig.fromYaml[K])
    )

}

class EmbedderCombiner[T](conf:EmbedderCombinerConfig[T])(implicit model: ParameterCollection) extends Embedder[T] {

  private val compressor   : Layer              = SingleLayer.compressor(conf.subEmbedderConfigs.map{_.outDim}.sum, conf.outDim)
  private val subEmbedders : List[Embedder[T]]  = conf.subEmbedderConfigs.map(_.construct())
  private val drop         : Dropout            = Dropout(conf.dropProb)

  override val outDim: Int = conf.outDim

//  override def apply(x: T): Expression =
//    drop(compressor(concatSeq(subEmbedders.map(e => e(x)))))

  override def apply(xs: List[T]): Expression =
    drop(compressor(concatSeq(subEmbedders.map(e => e(xs)))))

}
