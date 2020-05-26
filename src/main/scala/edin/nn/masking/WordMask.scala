package edin.nn.masking

import edu.cmu.dynet.Expression
import edin.nn.DyFunctions._

object WordMask{

  def apply(isMaskedElement:Array[Boolean]) : WordMask = new WordMask(isMaskedElement)

  def totallyUnmasked : WordMask = new WordMask(isMaskedElement = null)

  def padding[A](sents: List[List[A]]) : List[List[Option[A]]] = {
    val maxLen = sents.map(_.size).max
    val padded = for{
      sent    <- sents
      prefix   = sent.map(Some(_))
      postfix  = List.fill(maxLen-sent.size)(None)
    } yield prefix ++ postfix
    padded
  }

  def fromSents(sents:List[List[Array[Float]]]) : (List[Expression], List[WordMask]) = {
    val padded = padding(sents)
    val res = padded.transpose.map(fromWordsVertical)
    (res.map(_._1), res.map(_._2))
  }

  def fromWordsVertical(vertical:List[Option[Array[Float]]]) : (Expression, WordMask) = {
    // it doesn't make sense to create a version for when vertical.forall(_.isEmpty) because
    // then we don't know the dim to cannot create the return expression
    val expExample = vertical.find(_.nonEmpty).get.get
    val currDim    = expExample.length
    val mask       = new WordMask(isMaskedElement = vertical.map(_.isEmpty).toArray)
    val exp        = batchVector(vertical.map(_.getOrElse(Array.fill(currDim)(0f))))
    (exp, mask)
  }

}

class WordMask private (val isMaskedElement:Array[Boolean]) {

  private[masking] val totallyUnmasked = isMaskedElement==null || isMaskedElement.forall(!_)
  private[masking] lazy val maskOne    = batchScalar(isMaskedElement.map(if(_) 1f else 0f))
  private[masking] lazy val maskZero   = 1-maskOne

  def fillMask(exp:Expression, filler:Expression) : Expression =
    if(totallyUnmasked)
      exp
    else
      exp*maskZero + filler*maskOne

  def extractWordValuesVertical(exp:Expression) : List[Option[Array[Float]]] = {
    val values = exp.toBatchArray1d
    if(totallyUnmasked){
      values.toList.map(Option(_))
    }else{
      for(i <- isMaskedElement.indices.toList)
        yield if(isMaskedElement(i)) None else Some(values(i))
    }
  }

  def extractWord(exp:Expression, sentId:Int) : Option[Expression] =
    if(totallyUnmasked || !isMaskedElement(sentId))
      Some(exp.pickBatchElem(sentId))
    else
      None

  override def toString: String = "WordMask("+isMaskedElement.mkString(" ")+")"

}

