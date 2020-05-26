package edin.constituency.semirings

import collection.mutable.{PriorityQueue => MutPriorityQueue}
import edin.constituency.grammar.{BackPointer, BinaryBackPointer, Item, UnaryBackPointer}

sealed trait KBestState[I <: Item[I]]
sealed trait KBestStateEdge[I <: Item[I]] extends KBestState[I]{
  val currentBestScore : Double
  val currentBestNode : DeductionTree[I]
  def getNeighbours : List[KBestStateEdge[I]]
}
private case class KBestStateEdgeUnary[I <: Item[I]](
                                                      children:List[(Double, DeductionTree[I])],
                                                      item:I,
                                                      edge:BackPointer[I]
                                                    ) extends KBestStateEdge[I] {
  override val currentBestNode : DeductionTree[I] = DeductionTreeNonTerm(item, edge, List(children.head._2))
  override val currentBestScore: Double = children.head._1 + edge.score
  override def getNeighbours : List[KBestStateEdge[I]] =
    if(children.nonEmpty)
      this.copy[I](children = children.tail).asInstanceOf[KBestStateEdge[I]]::Nil
    else
      Nil
}

private case class KBestStateEdgeBinary[I <: Item[I]](
                                                       children1:List[(Double, DeductionTree[I])],
                                                       children2:List[(Double, DeductionTree[I])],
                                                       item:I,
                                                       edge:BackPointer[I]
                                                     ) extends KBestStateEdge[I] {
  override val currentBestNode : DeductionTree[I] = DeductionTreeNonTerm(item, edge, List(children1.head._2, children2.head._2))
  override val currentBestScore: Double = children1.head._1 + children2.head._1 + edge.score
  override def getNeighbours : List[KBestStateEdge[I]] =
    if(children1.nonEmpty && children2.nonEmpty)
      this.copy(children1 = children1.tail).asInstanceOf[KBestStateEdge[I]]::
        this.copy(children2 = children2.tail).asInstanceOf[KBestStateEdge[I]]::
        Nil
    else
      Nil
}
private case class KBestStateItem[I <: Item[I]](kBest:List[(Double, DeductionTree[I])]) extends KBestState[I]

sealed trait DeductionTree[I <: Item[I]]
case class DeductionTreeNonTerm[I <: Item[I]](item:I, bp:BackPointer[I], children:List[DeductionTree[I]]) extends DeductionTree[I]
case class DeductionTreeTerm[I <: Item[I]](item:I) extends DeductionTree[I]


/**
  * Algorithm 2 from Huang & Chiang Better k-best Parsing
  */
class KBest[I <: Item[I]](k:Int) extends SemiRing[I, KBestState[I]] {

  override def _1: KBestState[I] = ???

  override def ⨁(vs: Seq[KBestState[I]], item:I): KBestState[I] = vs match {
    case Seq() => KBestStateItem(List( (0, DeductionTreeTerm(item)) ))
    case xs =>
      val q = MutPriorityQueue.empty[KBestStateEdge[I]](
        Ordering.by((_: KBestStateEdge[I]).currentBestScore).reverse
      )
      for(v <- bestKOrdering(xs.asInstanceOf[Seq[KBestStateEdge[I]]])){
        q.enqueue(v)
      }

      var output = List[(Double, DeductionTree[I])]()
      var extracted = 0

      while(q.nonEmpty && extracted<k){
        val next = q.dequeue()
        output ::= (next.currentBestScore, next.currentBestNode)
        for(neighbour <- next.getNeighbours){
          q.enqueue(neighbour)
        }
        extracted += 1
      }

      KBestStateItem(output)
  }

  override def ⨂(vs: Seq[KBestState[I]], bp: BackPointer[I], item: I): KBestState[I] = (bp, vs) match {
    case (UnaryBackPointer(_, _,     _), Seq(childKBest:KBestStateItem[I])) =>
      KBestStateEdgeUnary(childKBest.kBest, item, bp)
    case (BinaryBackPointer(_, _, _, _), Seq(child1KBest:KBestStateItem[I], child2KBest:KBestStateItem[I])) =>
      KBestStateEdgeBinary(child1KBest.kBest, child2KBest.kBest, item, bp)
  }

  private def bestKOrdering(xs : Seq[KBestStateEdge[I]]) : List[KBestStateEdge[I]] = {
    val q = MutPriorityQueue.empty[KBestStateEdge[I]](
      Ordering.by((_: KBestStateEdge[I]).currentBestScore).reverse
    )
    for(x <- xs){
      q.enqueue( x )
      if(q.size > k){
        q.dequeue()
      }
    }
    var res = List[KBestStateEdge[I]]()
    while(q.nonEmpty){
      res ::= q.dequeue()
    }
    res
  }

}

