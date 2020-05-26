package edin.constituency.semirings

import edin.constituency.chart.Chart
import edin.constituency.chart.Query.Query
import edin.constituency.grammar.{BackPointer, BinaryBackPointer, Item, UnaryBackPointer}
import edin.constituency.deduction.ForwardChaining._

trait SemiRing[I <: Item[I], V <: AnyRef] {

  def _1 : V

  def ⨁(vs : Seq[V], item:I) : V

  def ⨂(vs : Seq[ V ], bp:BackPointer[I], item:I) : V




  final def ⨁(v1 : V, v2: V, item:I) : V = ⨁(Seq(v1, v2), item)

  final def ⨂(v1 : V, v2: V, bp:BackPointer[I], item:I) : V = ⨂(Seq(v1, v2), bp, item)

  final def forward(chart:Chart[I]) : Chart[V] = {
    val table = chart.newEmptyVersion[V]
    for(item <- topologicalSort(chart, findTopItems(chart))){
      val backPointerValues = for {
        bp          <- item.backpointers
        antecedents =  bp.antecedents
        vs          =  for(a <- antecedents) yield table(a.signature)
        bpValues    =  ⨂(vs, bp, item)
      }
        yield bpValues
      table(item.signature) = ⨁(backPointerValues, item)
    }
    table
  }

  final def backward(chart:Chart[I], forward:Chart[V], rootQuery:Query) : Chart[V] = {
    val table = chart.newEmptyVersion[V]
    val roots = chart.query(rootQuery)
    for(root <- roots)
      table(root.signature) = _1

    for{
      item <- topologicalSort(chart, findTopItems(chart)).reverse
      bp   <- item.backpointers
    } {
      bp match {
        case UnaryBackPointer(_, itemChild, _) =>
          table(itemChild.signature) = ⨁(table(itemChild.signature), ⨂(table(item.signature), _1, bp, item), item)
        case BinaryBackPointer(_, itemChild1, itemChild2, _) =>
          table(itemChild1.signature) = ⨁(table(itemChild1.signature), ⨂(table(item.signature), forward(itemChild2.signature), bp, item), item)
          table(itemChild2.signature) = ⨁(table(itemChild2.signature), ⨂(table(item.signature), forward(itemChild1.signature), bp, item), item)
      }
    }
    table
  }

}

