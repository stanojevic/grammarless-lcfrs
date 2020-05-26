package edin.constituency.semirings

import java.lang.{Double => JDouble}

import edin.constituency.grammar.{BackPointer, Item}

class Inside[I <: Item[I]] extends SemiRing[I, JDouble] {

  override def _1: JDouble = JDouble.valueOf(0d)

  override def ⨁(vs: Seq[JDouble], item:I): JDouble = logSumExp(vs.map(_.toDouble))

  override def ⨂(vs: Seq[JDouble], bp: BackPointer[I], item: I): JDouble = vs.map(_.toDouble).sum + bp.score

  private def logSumExp(a:Double, b:Double) : Double =
    if(a.isNegInfinity){
      b
    }else if(b.isNegInfinity){
      a
    }else{
      val x = math.max(a, b)
      val y = math.min(a, b)
      x + math.log1p(math.exp(y-x))
    }

  private def logSumExp(logProbs:Seq[Double]) : Double =
    if (logProbs.tail.isEmpty)
      logProbs.head
    else
      logProbs.reduce(logSumExp)
}
