package edin.algorithms


/**
  * A version of union-find algorithm explained in section 2.2 of "A Persistent Union-Find Data Structure" by Conchon and Filliatre
  * The reason why I don't use a faster and more complicated version in later sections is because that one requires knowing
  * all the elements ahead of time while this implementation is more flexible.
  *
  * This data structure contains information about relations between multiple disjoint sets.
  *
  * The implementation is only asymptotic if used ephemerally or if the same version of the
  * structure is repeatedly used with limited backtracking (this is true for all versions in the paper).
  *
  * With union-by-rank & path-compression heuristics:
  * If used ephimerally for m operations on n elements it takes O(m log(n))
  * log(n) is caused by immutable maps
  * Scala's Maps are "effectivelly constant" but that's kind of contraversial
  *
  * With union-by-rank heuristic alone:
  * It will probably behave better in non-ephimeral use cases where, I think, it has complexity of O(m*log(n)*log(n))
  *
  */

class UnionFindPersistent[E] private (
                              private var rank: Map[E, Int],
                              private var parent: Map[E, E],
                              withPathCompression : Boolean
                            ){

  def contains(e:E) : Boolean = rank.contains(e)

  def size : Int = rank.size

  def addSelfSet(e:E) : UnionFindPersistent[E] = new UnionFindPersistent[E](rank + (e -> 0), parent + (e->e), withPathCompression)

  def unionSafe(e1:E, e2:E) : UnionFindPersistent[E] = (contains(e1), contains(e2)) match {
    case (true , true ) => union(e1, e2)
    case (true , false) => addSelfSet(e2).union(e1, e2)
    case (false, true ) => addSelfSet(e1).union(e1, e2)
    case (false, false) => addSelfSet(e1).addSelfSet(e2).union(e1, e2)
  }

  override def toString: String = "UF("+parent.toList.map{case (x,y) => s"$x -> $y"}.mkString(" ")+")"

  def toList: List[(E, E)] = rank.keys.toList.map(e => (e, findSetRep(e)))

  def union(e1:E, e2:E) : UnionFindPersistent[E] = {
    val end1 = findSetRep(e1)
    val end2 = findSetRep(e2)
    if(end1 == end2){
      this
    }else{
      val rank1 = rank(end1)
      val rank2 = rank(end2)
      if(rank1 > rank2)
        new UnionFindPersistent[E](
          rank   = rank,
          parent = parent + (end2 -> end1),
          withPathCompression
        )
      else if(rank1 < rank2)
        new UnionFindPersistent[E](
          rank   = rank,
          parent = parent + (end1 -> end2),
          withPathCompression
        )
      else
        new UnionFindPersistent[E](
          rank   = rank   + (end1 -> (rank1+1)),
          parent = parent + (end2 -> end1     ),
          withPathCompression
        )
    }
  }

  def get(e:E) : Option[E] = if(contains(e)) Some(findSetRep(e)) else None

  def findSetRep(e:E) : E = parent(e) match {
    case `e` =>
      e
    case y =>
      val end = findSetRep(y)
      if(withPathCompression)
        parent += e -> end
      end
  }

}

object UnionFindPersistent{

  def apply[E](withPathCompression:Boolean) : UnionFindPersistent[E] = new UnionFindPersistent[E](Map(), Map(), withPathCompression)

  def apply[E](withPathCompression:Boolean, selfSetElems:E*) : UnionFindPersistent[E] = fromSeq(selfSetElems, withPathCompression)

  def fromSeq[E](selfSetElems:Seq[E], withPathCompression:Boolean) : UnionFindPersistent[E] = selfSetElems.foldLeft(UnionFindPersistent[E](withPathCompression))(_ addSelfSet _)

  def main(args:Array[String]) : Unit = {
    val uf  = UnionFindPersistent(withPathCompression = false, 1 ,3 , 4)
    val uf2 = uf.union(3, 4)
    val uf3 = uf2.union(4, 1)
    uf.findSetRep(4)
    println(uf3)
    println( uf.contains(4) )
    println( uf.contains(2) )
    println("Hello")
  }

}
