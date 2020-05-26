package edin.nn.attention

import edu.cmu.dynet.{Expression, ParameterCollection}
import edin.nn.DyFunctions._
import edin.general.RichScala._

class WordPairScorerBiAffine(val config:WordPairScorerConfig)(implicit model:ParameterCollection) extends WordPairScorer {

  override val labels: Int = config.labels

  import config._

  private val dimX = dim+1
  private val dimY = dim+1

  private val W = addParameters((dimY*labels, dimX), initAround(0))

  override def apply(x:Expression, y:Expression) : Expression = {
    val n = x.cols
    val b = x.batchSize

    val X = concat(x, ones(1, n))
    val Y = concat(y, ones(1, n))

    if (labels == 1){
      Y.T * W * X
    }else{
      (  W*X                             ) |>
      (  _ reshape ((dimY, labels*n), b) ) |>
      (  Y.T*_                           ) |>
      (  _ reshape ((n, labels, n), b)   )
    }
  }

  override def apply(xs: List[Expression], ys: List[Expression]): Expression =
    apply(concatCols(xs:_*), concatCols(ys:_*))

}
