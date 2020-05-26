package edin.constituency.chart

import scala.language.implicitConversions

object Query{

  type Address = List[Int]
  type Query   = List[ChartCond]

  sealed trait Term
  case class Var (x: Symbol) extends Term
  case class TInt(x: Int   ) extends Term

  sealed trait ChartCond
  case class  Pred(f:Int=>Boolean) extends ChartCond
  case object Star                 extends ChartCond
  case class  EqCond(i:Int)        extends ChartCond

  def queryMatch(address:List[Int], query:Query) : Boolean = (address, query) match {
    case (Nil  , Nil          )           => true
    case (a::as, Pred(f)  ::qs) if f(a)   => queryMatch(as, qs)
    case (a::as, Star     ::qs)           => queryMatch(as, qs)
    case (a::as, EqCond(i)::qs) if a == i => queryMatch(as, qs)
    case _                                => false
  }

  // this is just to make things easier to use
  implicit def f2FuncCond(f:Int=>Boolean) : ChartCond = Pred(f)
  implicit def int2EqCond(i:Int         ) : ChartCond = EqCond(i)
  implicit def intList2EqConds(is:List[Int]) : List[ChartCond] = is.map(EqCond)
  val __ : ChartCond = Star
  val ?  : ChartCond = Star

//  def main(args:Array[String]) : Unit = {
//
//    val m = IntMapTinySize(2 -> "aha", 4 -> "so this is it", 1 -> "ok")
//    val m2 = IntMapFixedSize(6, 2 -> "aha", 4 -> "so this is it", 1 -> "ok")
//    println(m)
//    println(m2)
//
//    val b = BitSetC(2, 0, 1, 63, 70)
//    println(b.toList)
//
//    type Cell = String
//    var chart: ChartNode[Cell] = Empty()
//    val a1 = List(2, 3, 1, 0)
//    val a2 = List(2, 4, 2)
//    chart = chart.add(a1, "Blah")
//    chart = chart.add(a1, "Blah2")
//    chart = chart.add(a2, "Blah3")
//    println(chart)
//    println(chart.find(a1).toList)
//    println(chart.find(List(?, Pred(_<=3), ??)).toList)
//    println(chart.find(?, 3, ?, 0).toList)
//    println(chart.find(a2).toList)
//
//
//    println(b)
//    println(b.count_elems_less_than(63))
//
//  }

}
