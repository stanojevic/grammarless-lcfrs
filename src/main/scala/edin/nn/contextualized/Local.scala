package edin.nn.contextualized

import edin.nn.DyFunctions._
import edin.nn.DyFunctions
import edin.nn.embedder.{Embedder, EmbedderConfig}
import edin.nn.masking.WordMask
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Dim, Expression, ParameterCollection}

/**
  * This is just a wrapper to the most standard embedding of words by
  * ether lookup table, char-lstm, combination of them, position emb etc.
  * basically anything that doesn't look at the global context but only
  * the local word.
  */

sealed case class LocalConfig[T](
                                  embConf:EmbedderConfig[T]
                                ) extends SequenceEmbedderIncrementalConfig[T] {
  override val outDim: Int = embConf.outDim
  override def construct()(implicit model: ParameterCollection): SequenceEmbedderIncremental[T] = new Local[T](this)
}

object LocalConfig{

  def fromYaml[K](conf:YamlConfig) : SequenceEmbedderIncrementalConfig[K] =
    LocalConfig[K](
      embConf = EmbedderConfig.fromYaml(conf("embedder-conf"))
    )

}

class Local[T](config: LocalConfig[T])(implicit model: ParameterCollection) extends SequenceEmbedderIncremental[T] {

  private val embedder : Embedder[T] = config.embConf.construct()

  override def zeros: Expression = DyFunctions.zeros(config.embConf.outDim)

  override def initState(): IncrementalEmbedderState[T] = new SequenceEmbedderStandardState[T](embedder)

}

class SequenceEmbedderStandardState[T](e:Embedder[T]) extends IncrementalEmbedderState[T] {

  override def nextStateAndEmbedBatch(xs: List[Option[T]]): (IncrementalEmbedderState[T], Expression, WordMask) = {
    val (exp, mask) = if(xs.forall(_.nonEmpty)){
      (e(xs.map(_.get)), WordMask.totallyUnmasked)
    }else{
      xs.find(_.nonEmpty) match {
        case Some(Some(expExample)) =>
          val xs2 = xs.map{
            case Some(x) => x
            case None => expExample
          }
          val exp = e(xs2)
          val mask = WordMask(xs.map(_.isEmpty).toArray)
          (exp, mask)
        case _ =>
          val exp = zeros(Dim(List(e.outDim), xs.size))
          val mask = WordMask(Array.fill(xs.size)(true))
          (exp, mask)
      }
    }
    (this, exp, mask)
  }

}

