package edin.constituency.chart

import edin.constituency.chart.Query.Query

/**
  * This is basically a naive associative list that supports a lookup using a query vector
  */
class ChartList[B] extends Chart[B]{

  override protected val name: String = "ChartList"

  private var allEntries = List[(List[Int], B)]()

  override def query(query: Query): Stream[B] =
    for { (a, b) <- allEntries.toStream if Query.queryMatch(a, query)}
      yield b

  // this method assumes that bucket is not already present
  override def +=(kv: (List[Int], B)): ChartList.this.type = {
    if(allEntries.forall(_._1 != kv._1))
      allEntries ::= kv
    this
  }

  override def iterator: Iterator[(List[Int], B)] =
    allEntries.iterator

  override def -=(key: List[Int]): ChartList.this.type = {
    allEntries = allEntries.filterNot(_._1 == key)
    this
  }

  override def newEmptyVersion[V <: AnyRef]: Chart[V] = new ChartList()
}


