package edin.constituency.semirings

import edin.constituency.grammar.{BackPointer, Item}

class Counting[I <: Item[I]] extends SemiRing[I, Integer] {

  override def _1: Integer = 1

  override def ⨁(vs: Seq[Integer], item:I): Integer = vs.map(_.toInt).sum

  override def ⨂(vs: Seq[Integer], bp: BackPointer[I], item: I): Integer = vs.map(_.toInt).product

}

