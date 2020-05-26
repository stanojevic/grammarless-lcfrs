package edin.constituency.agenda

import edin.constituency.chart.{Chart, ChartOptimized}
import edin.constituency.grammar.Item

class AgendaUnique[I <: Item[I]](uniquenessCache: Chart[Option[I]]) extends Agenda[I] {

  require(!uniquenessCache.isInstanceOf[ChartOptimized[_]], "chart-optimized is not the right choice")

  private var ssize = 0
  private var store = List[I]()

  override def head: I = store.head

  override def pop(): I = {
    val i = store.head
    store = store.tail
    ssize -= 1
    uniquenessCache(i.signature) = None // this is better than chart.remove because remove may be too expensive compared to overwrite
    i
  }

  override def add(i: I): Unit = uniquenessCache.get(i.signature) match {
    case Some(Some(item)) =>
      for(b <- i.backpointers)
        item.addBackPointer(b)
    case _ =>
      store ::= i
      uniquenessCache(i.signature) = Some(i)
      ssize += 1
  }

  override def size: Int = ssize

  override def toStream: Stream[I] = store.toStream

}
