package edin.constituency.agenda

import edin.constituency.grammar.Item

trait Agenda[I <: Item[I]] {

  // AgendaPriority can override it so it could optimize using heapification
  def addAxioms(axioms: Seq[I]): Unit = {
    require(isEmpty)
    for(axiom <- axioms){
      add(axiom)
    }
  }

  def head : I

  def pop() : I

  def add(i:I) : Unit

  def size : Int

  def toStream : Stream[I]

  final def isEmpty : Boolean =
    size==0

  final def nonEmpty : Boolean =
    !isEmpty

  override final def toString: String =
    "Agenda(\n"+
    toStream.map(i => s"\t$i").mkString("\n")+
    "\n)"

}
