package edin.constituency.grammar

import edin.constituency.chart.Query.Query

sealed trait BackPointer[I <: Item[I]] {
  def antecedents : Seq[I] = this match {
    case BinaryBackPointer(_, item1, item2, _) => List(item1, item2)
    case UnaryBackPointer(_, item, _)          => List(item)
  }
  def score : Double = this match {
    case UnaryBackPointer (_, _,    score) => score
    case BinaryBackPointer(_, _, _, score) => score
  }
}
case class BinaryBackPointer[I <: Item[I]](rule:BinaryRule[I], itemA:I, itemC:I, sscore:Double) extends BackPointer[I]
case class UnaryBackPointer [I <: Item[I]](rule: UnaryRule[I],          itemC:I, sscore:Double) extends BackPointer[I]

trait Item[I <: Item[I]] { self : I => // "self" is new "this" that has the type of the subtype

  private var binaryBackPointers = List[BinaryBackPointer[I]]()
  private var unaryBackPointers  = List[UnaryBackPointer [I]]()

  def addBackPointers : Seq[BackPointer[I]] => Unit = _ foreach addBackPointer

  def addBackPointer : BackPointer[I] => Unit = {
    case b: BinaryBackPointer[_] => binaryBackPointers ::= b.asInstanceOf[BinaryBackPointer[I]]
    case b: UnaryBackPointer[_]  => unaryBackPointers  ::= b.asInstanceOf[ UnaryBackPointer[I]]
  }

  def backpointers : Stream[BackPointer[I]] = unaryBackPointers.toStream ++ binaryBackPointers.toStream

  var bestScore : Double // includes only best inside score in case of A* or Knuth search

  var priority : Double // includes inside+outside score in case of A* search

  def signature : List[Int]

}

trait UnaryRule[I <: Item[_]]{

  def inferences(itemA:I) : Stream[I]

}

trait BinaryRule[I <: Item[_]]{

  def inferences(itemA:I, itemC:I) : Stream[I]

  def queryVectors(itemA:I) : Seq[Query]

}

trait GrammarState[I <: Item[_], URule <: UnaryRule[I], BRule <: BinaryRule[I]] {

  def moveToNextBatchElem() : Unit

  def axioms : List[I]

  def goalQuery : Query

  def scoreBinary(rule:BRule, itemA:I, itemC:I) : Double

  def scoreUnary(rule:URule, itemA:I) : Double

  def lookupBinaryRules(itemA:I) : List[BRule]

  def lookupUnaryRules(itemA:I) : List[URule]

}

trait Grammar[I <: Item[I], URule <: UnaryRule[I], BRule <: BinaryRule[I]] {

  // used for precomputing contextualised embeddings or masking rare words
  def initParsingBatch(sents: List[List[String]]) : GrammarState[I, URule, BRule]

}
