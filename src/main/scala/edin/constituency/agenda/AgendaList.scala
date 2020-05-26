package edin.constituency.agenda

import edin.constituency.grammar.Item

class AgendaList[I <: Item[I]] extends Agenda[I]{

  private var ssize = 0
  private var store = List[I]()

  override def size: Int =
    ssize

  override def pop(): I = {
    val item = store.head
    store = store.tail
    ssize -= 1
    item
  }

  override def add(i: I): Unit = {
    ssize += 1
    store ::= i
  }

  override def toStream: Stream[I] =
    store.toStream

  override def head: I = store.head

}

