package edin.constituency.deduction

import edin.constituency.agenda.Agenda
import edin.constituency.chart.Chart
import edin.constituency.grammar.{BinaryRule, GrammarState, Item, UnaryRule}

object ForwardChaining {

  def closure[I <: Item[I], URule <: UnaryRule[I], BRule <: BinaryRule[I]]
                                (
                                 agenda            : Agenda[I],
                                 chart             : Chart[I],
                                 grammarState      : GrammarState[I, URule, BRule],
                                 stopWhenGoalFound : Boolean = false,
                                 maxItemsLimit     : Long    = Long.MaxValue
                                // TODO time limit
                                ) : Boolean = {
    def chartContainsGoal : Boolean = chart.query(grammarState.goalQuery).nonEmpty

    agenda addAxioms grammarState.axioms

    var processedItems = 0l
    while(agenda.nonEmpty && !(stopWhenGoalFound && chartContainsGoal) && (processedItems < maxItemsLimit)){
      val itemA = agenda.pop()
      processedItems+=1

      chart.get(itemA.signature) match {
        case Some(itemC) =>
          // the item is already in the chart so just add its backpointers to the chart element
          itemC addBackPointers itemA.backpointers
        case None =>
          // add trigger to the chart
          chart += itemA.signature -> itemA

          // add unary consequents to the agenda
          for{
            unaryRule  <- grammarState lookupUnaryRules itemA
            consequent <- unaryRule    inferences       itemA
          } agenda add consequent

          // add binary consequents to the agenda
          for{
            binaryRule <- grammarState lookupBinaryRules itemA
            query      <- binaryRule   queryVectors      itemA
            itemC      <- chart        query             query
            consequent <- binaryRule.inferences(itemA, itemC)
          } agenda add consequent

      }
    }

    chartContainsGoal
  }

  def findTopItems[I <: Item[I]](chart : Chart[I]) : Stream[I] = {
    val visitedCache = chart.newEmptyVersion[I]

    for{
      item        <- chart.values
      backpointer <- item.backpointers
      antecedent  <- backpointer.antecedents
    } visitedCache(antecedent.signature) = antecedent

    chart.values.toStream.filterNot(i => visitedCache.contains(i.signature))
  }

  def topologicalSort[I <: Item[I]](
                       chart             : Chart[I],
                       rootItems         : Seq[I],
                     ) : Stream[I] = {
    val visitedCache = chart.newEmptyVersion[I]
    val topsort = collection.mutable.ArrayBuffer[I]()

    def visit(node:I) : Unit =
      if(!(visitedCache contains node.signature)){
        visitedCache(node.signature) = node
        node.backpointers.flatMap(_.antecedents).foreach(visit)
        topsort += node
      }

    rootItems.foreach(visit)
    topsort.toStream
  }

}
