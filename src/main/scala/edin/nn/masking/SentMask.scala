package edin.nn.masking

import edu.cmu.dynet.{Dim, Expression}
import edin.nn.DyFunctions._
import edin.nn.DynetSetup
import scala.collection.mutable.{Map => MutMap}

object SentMask{

  def main(args:Array[String]) : Unit = {
    DynetSetup.init_dynet()

    val sm = SentMask(List(WordMask(Array(false, false, false)), WordMask(Array(true, false, false))))

    val e = Expression.randomUniform(((4, 2), 9), -1, 1)

    e.printWithVals()
    sm.fillMask(e, ones(4)*7, 3).printWithVals()
//    sm.fillMask(e, ones(4)*7).printWithVals()
  }

  def apply(wordMasks:List[WordMask]) : SentMask =
    new SentMask(wordMasks)

}

class SentMask private (val wordMasks:List[WordMask]){

  private val totallyUnmasked = wordMasks.forall(_.totallyUnmasked)
  private val maskOneMap  = MutMap[(Int, Int), Expression]()
  private val maskZeroMap = MutMap[(Int, Int), Expression]()
  private val sentLen = wordMasks.size
  private lazy val batchSize    = wordMasks.find(!_.totallyUnmasked).get.isMaskedElement.length
  private lazy val maskNoneWord  = zeros(Dim(List(1), batchSize))
  private lazy val maskOnePrototype    = concatCols(wordMasks.map(m => if(m.totallyUnmasked) maskNoneWord else m.maskOne ):_*)
  private lazy val maskZeroPrototype   = 1-maskOnePrototype

  def fillMask(exp:Expression, fillerVec:Expression, expansion:Int=1) : Expression = {
    assert(exp.cols == wordMasks.size)
    if(totallyUnmasked){
      exp
    }else{
      val (maskOne, maskZero) = getMasks(exp.rows, expansion)
      exp⊙maskZero + fillerVec⊙maskOne
    }
  }

  private def getMasks(rowDim:Int, expansion:Int) : (Expression, Expression) = {
    if(! maskOneMap.contains((rowDim, expansion))){
      if(expansion == 1){
        val maskOne = ones(rowDim, 1)*maskOnePrototype
        maskOneMap((rowDim, 1))  = maskOne
        maskZeroMap((rowDim, 1)) = 1-maskOne
      }else{
        assert(expansion>1)
        val (mOne, _) = getMasks(rowDim, 1)
        // TODO Mask extension by orthogonal multiplication instead of concatenation ---V bellow
        val maskOne = concatCols((for(_ <- 0 until expansion) yield mOne) :_*).reshape((rowDim, sentLen), batchSize*expansion)
        maskOneMap((rowDim, expansion)) = maskOne
        maskZeroMap((rowDim, expansion)) = 1-maskOne
      }
    }
    (maskOneMap((rowDim, expansion)), maskZeroMap((rowDim, expansion)))
  }

  override def toString: String = "---------SentMask------------\n"+maskOnePrototype.toStr

}

