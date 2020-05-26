package edin.nn.embedder

import edin.nn.contextualized.LocalConfig
import edin.nn.model.Any2Int
import edin.nn.sequence.RecurrentNN
import edin.nn.DyFunctions._
import edu.cmu.dynet.{Expression, ParameterCollection}

final case class EmbedderSingleVectorForSeqConfig[W, C](
                                                         outDim        : Int,
                                                         breakToPieces : W => List[C],
                                                         piece2Int     : Any2Int[C]
                                                       ) extends EmbedderConfig[W] {

  def construct()(implicit model: ParameterCollection) = new EmbedderSingleVectorForSeq[W, C](this)

}

class EmbedderSingleVectorForSeq[W, C] (config: EmbedderSingleVectorForSeqConfig[W, C])(implicit model: ParameterCollection) extends Embedder[W] {

  override val outDim: Int = config.outDim

  private val seqEmbedderConfig = LocalConfig( EmbedderStandardConfig(
    s2i    = config.piece2Int,
    outDim = outDim
  ) )
  private val seqEmbedder = seqEmbedderConfig.construct()
  private val encoderFwd = RecurrentNN.singleFactory("lstm", seqEmbedderConfig.outDim, seqEmbedderConfig.outDim/2, 0f, "variational", false)
  private val encoderBck = RecurrentNN.singleFactory("lstm", seqEmbedderConfig.outDim, seqEmbedderConfig.outDim/2, 0f, "variational", false)

  override def apply(xs: List[W]): Expression = {

    val parts = xs map config.breakToPieces
    val (bckE, bckM) = seqEmbedder.transduceBatch(parts)
    val b = encoderBck.transduceBackward(bckE, bckM).head
    val (fwdE, fwdM) = seqEmbedder.transduceBatch(parts.map(_.reverse))
    val f = encoderFwd.transduceBackward(fwdE, fwdM).head

    b concat f
  }

}
