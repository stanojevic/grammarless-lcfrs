package edin.constituency.chart

import edin.constituency.chart.Query.{ChartCond, EqCond, Pred, Query, Star}

import scala.annotation.tailrec

class ChartTable[B](dimensions:List[Int], arrayBased:Boolean) extends Chart[B]{

  override protected val name: String = "ChartTable"

  private val allEntries = if(arrayBased) new ArrayMap[B](dimensions) else scala.collection.mutable.Map[List[Int], B]()

  private def genCandidates(query:Query, dims:List[Int]) : Stream[List[Int]] = (query, dims) match {
    case (List(q), List(d)) =>
      singleDimOptions(q, d).map(_::Nil)
    case (q::qs, d::ds) =>
      val as = genCandidates(qs, ds)
      val is = singleDimOptions(q, d)
      for{
        a <- as
        i <- is
      }
        yield i::a
  }

  private def singleDimOptions(q:ChartCond, d:Int) : Stream[Int] = q match {
    case Star      => (0 until d).toStream
    case EqCond(i) => Stream(i)
    case Pred(p)   => (0 until d).toStream.filter(p)
  }

  override def query(query: Query): Stream[B] = genCandidates(query, dimensions).flatMap(allEntries.get)

  override def +=(kv: (List[Int], B)): ChartTable.this.type = {
    allEntries(kv._1) = kv._2
    this
  }

  override def iterator: Iterator[(List[Int], B)] = allEntries.iterator

  override def -=(key: List[Int]): ChartTable.this.type = {
    allEntries.remove(key)
    this
  }

  override def newEmptyVersion[V <: AnyRef]: Chart[V] = new ChartTable(dimensions, arrayBased)
}

private class ArrayMap[B](dimensions:List[Int]) extends collection.mutable.Map[List[Int], B]{

  private def constructArray(dimensions:List[Int]) : Array[AnyRef] = dimensions match {
    case List(x) => Array.ofDim[AnyRef](x)
    case x::xs   =>
      val res = Array.ofDim[AnyRef](x)
      for(i <- res.indices){
        res(i) = constructArray(xs)
      }
      res
  }

  @tailrec
  private def lookupArray(address:List[Int], tree:Array[AnyRef]) : AnyRef = address match {
    case List(x) => tree(x)
    case x::xs   => lookupArray(xs, tree(x).asInstanceOf[Array[AnyRef]])
  }

  @tailrec
  private def putInArray(address:List[Int], tree:Array[AnyRef], value:AnyRef) : Unit = address match {
    case List(x) => tree(x) = value
    case x::xs   => putInArray(xs, tree(x).asInstanceOf[Array[AnyRef]], value)
  }

  private val arrayMap = constructArray(dimensions)

  override def +=(kv: (List[Int], B)): ArrayMap.this.type = {
    require(kv._1.size == dimensions.size)
    putInArray(kv._1, arrayMap, kv._2.asInstanceOf[AnyRef])
    this
  }

  override def -=(key: List[Int]): ArrayMap.this.type = {
    var curr: Array[AnyRef] = arrayMap
    for(k <- key.init){
      curr = curr(k).asInstanceOf[Array[AnyRef]]
    }
    curr(key.last)=null
    this
  }

  override def get(key: List[Int]): Option[B] =
    Option(lookupArray(key, arrayMap)).map(_.asInstanceOf[B])

  private val iteratorRec : AnyRef => Stream[(List[Int], B)] = {
    case tree:Array[AnyRef] =>
      for{
        i <- tree.indices.toStream
        if tree(i) != null
        sub <- iteratorRec(tree(i))
      }
        yield (i :: sub._1, sub._2)
    case null =>
      Stream.empty
    case stuff =>
      Stream((List(), stuff.asInstanceOf[B]))
  }

  override def iterator: Iterator[(List[Int], B)] = iteratorRec(arrayMap).iterator

}
