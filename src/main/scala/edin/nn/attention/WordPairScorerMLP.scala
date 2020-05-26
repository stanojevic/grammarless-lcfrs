package edin.nn.attention

import edin.nn.layers.MLPConfig
import edin.nn.DyFunctions._
import edu.cmu.dynet.{Expression, ParameterCollection}

import scala.collection.mutable.ArrayBuffer

class WordPairScorerMLP(val config:WordPairScorerConfig)(implicit model:ParameterCollection) extends WordPairScorer {

  override val labels: Int = config.labels

  private val mlp = MLPConfig(
    activations = List("relu", "linear"),
    sizes = List(2*config.dim, config.dim, config.labels)
  ).construct()

  override def apply(x: Expression, y: Expression): Expression = {
    val xLen = x.cols
    val yLen = y.cols
    val cols = ArrayBuffer[Expression]() // (n, L), b
    for(j <- 0 until yLen){
      val col = ArrayBuffer[Expression]() //  (1, L), b
      for(i <- 0 until xLen){
        col.append(mlp(concat(x.col(i), y.col(j))).T)
      }
      cols.append(concat(col:_*))
    }
    if(labels == 1){
      concatByDim(cols, d=1)
    }else{
      concatByDim(cols, d=2)
    }
  }

  override def apply(xs: List[Expression], ys: List[Expression]): Expression = {
    val xLen = xs.size
    val yLen = ys.size
    var cols = List[Expression]() // (n, L), b
    for(i <- 0 until xLen){
      var col = List[Expression]() //  (1, L), b
      for(j <- 0 until yLen){
        col ::= mlp(concat(xs(i), ys(j))).T
      }
      cols ::= concat(col.reverse:_*)
    }
    if(labels == 1){
      concatByDim(cols.reverse, d=1)
    }else{
      concatByDim(cols.reverse, d=2)
    }
  }

}
