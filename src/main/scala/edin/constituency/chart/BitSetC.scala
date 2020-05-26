package edin.constituency.chart

import scala.annotation.tailrec
import java.lang.Long.bitCount

/**
  * BitSet implementation that can store integers between 0 and 127
  * Compared to Scala's BitSet this one has an additional advantage of being
  * able to efficiently count elements smaller than some particular number.
  */
object BitSetC{
  def apply(xs:Int*) : BitSetC = fromSeq(xs)
  def fromSeq(xs:Seq[Int]) : BitSetC = xs.foldLeft(new BitSetC(0, 0))(_+_)
}

class BitSetC private(maskUpper:Long, maskLower:Long){

  @inline
  def -(i:Int) : BitSetC =
    if(i<64 && i>=0)
      new BitSetC(maskUpper, maskLower & (~(1 << i)))
    else if(i<128 && i>=0)
      new BitSetC(maskUpper & (~(1 << i)), maskLower)
    else
      sys.error(s"can't process integer $i because it's bigger than 128")

  @inline
  def +(i:Int) : BitSetC =
    if(i<64 && i>=0)
      new BitSetC(maskUpper, maskLower | (1l << i))
    else if(i<128 && i>=0)
      new BitSetC(maskUpper | (1l << (i-64)), maskLower)
    else
      sys.error(s"can't process integer $i because it's bigger than 128")

  @inline
  def contains(i:Int) : Boolean =
    if(i<64 && i>=0)
      (maskLower & (1l << i)) != 0
    else if(i<128 && i>=0)
      (maskUpper & (1l << (i-64))) != 0
    else
      sys.error(s"can't process integer $i because it's bigger than 128")

  @inline
  def count_elems_less_than(i:Int) : Int =
    if(i == 0)
      0
    else if(i<64 && i>=0)
      bitCount(maskLower & ((~0l) >>> ( 64-i)))
    else if(i<128 && i>=0)
      bitCount(maskUpper & ((~0l) >>> (128-i))) + bitCount(maskLower)
    else
      sys.error(s"can't process integer $i because it's bigger than 128")

  @inline
  def size : Int = bitCount(maskUpper) + bitCount(maskLower)

  // def toList : List[Int] = (0 until 128).filter(contains).toList
  def toList : List[Int] = extractNums(maskLower, 0, Nil).reverse ++ extractNums(maskUpper, 64, Nil).reverse

  @tailrec
  private def extractNums(mask:Long, start:Int, acc:List[Int]) : List[Int] =
    if(mask == 0)
      acc
    else if((mask & 1l) == 1l)
      extractNums(mask >>> 1, start+1, start::acc)
    else
      extractNums(mask >>> 1, start+1, acc)

  override def toString: String =
    "BitSet2My("+ (toList mkString ", ") +")"

}
