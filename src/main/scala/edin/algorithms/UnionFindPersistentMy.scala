package edin.algorithms


/**
  * This is my attempt to improve asymptothic runtime of persistent Union-Find algorithm
  * It is a very lazy version that does both union-by-rank and path-compression.
  *
  * It should be efficient to use even in non-ephimeral use cases which is not true
  * for UnionFind by Conchon and Filliatre.
  *
  * I belive its complexity is O(m log n) which is the best complexity one can
  * hope for in immutable setting.
  *
  * In practice though it doesn't seem to be super fast probably because of the
  * constants that come from laziness.
  *
  */
class UnionFindPersistentMy[E] private (
                                         private var rank: Map[E, Int],
                                         private var parent: KeyValueStore[E, E],
                                       ){

  def contains(e:E) : Boolean = rank.contains(e)

  def size : Int = rank.size

  def addSelfSet(e:E) : UnionFindPersistentMy[E] = new UnionFindPersistentMy[E](rank + (e -> 0), parent + (e, e))

  def findSetRep(e:E) : E = parent(e)

  def get(e:E) : Option[E] = if(contains(e)) Some(findSetRep(e)) else None

  def unionSafe(e1:E, e2:E) : UnionFindPersistentMy[E] = (contains(e1), contains(e2)) match {
    case (true , true ) => union(e1, e2)
    case (true , false) => addSelfSet(e2).union(e1, e2)
    case (false, true ) => addSelfSet(e1).union(e1, e2)
    case (false, false) => addSelfSet(e1).addSelfSet(e2).union(e1, e2)
  }

  def toList: List[(E, E)] = rank.keys.toList.map(e => (e, findSetRep(e)))

  override def toString: String = "UF("+parent.toList.map{case (x,y) => s"$x->$y"}.mkString(", ")+")"

  def union(e1:E, e2:E) : UnionFindPersistentMy[E] = {
    val end1 = findSetRep(e1)
    val end2 = findSetRep(e2)
    if(end1 == end2){
      this
    }else{
      val rank1 = rank(end1)
      val rank2 = rank(end2)
      if(rank1 > rank2)
        new UnionFindPersistentMy[E](
          rank   = rank,
          parent = parent.map(findSetRep) + (end2, end1)
        )
      else if(rank1 < rank2)
        new UnionFindPersistentMy[E](
          rank   = rank,
          parent = parent.map(findSetRep) + (end1, end2)
        )
      else
        new UnionFindPersistentMy[E](
          rank   = rank + (end1 -> (rank1+1)),
          parent = parent.map(findSetRep) + (end2, end1)
        )
    }
  }

}

object UnionFindPersistentMy{

  def apply[E]() : UnionFindPersistentMy[E] = new UnionFindPersistentMy[E](Map(), LazyMap())

  def apply[E](selfSetElems:E*) : UnionFindPersistentMy[E] = fromSeq(selfSetElems)

  def fromSeq[E](selfSetElems:Seq[E]) : UnionFindPersistentMy[E] = selfSetElems.foldLeft(UnionFindPersistentMy[E]())(_ addSelfSet _)

  def main(args:Array[String]) : Unit = {


    val start = System.currentTimeMillis()
    var ufos = List(UnionFindPersistentMy.fromSeq(0 until 10000))
//    var ufos = List(UnionFindPersistent.fromSeq(0 until 10000, withPathCompression = true))

    for(i <- 0 until 10000-1){
      ufos ::= ufos.head.union(i, i+1)
    }



//    for(moduo <- 2 until 50){
//      for(i <- 0 until 100){
//        if(i+moduo< 100){
//          ufos ::= ufos.head.union(i, i+moduo)
//        }
//      }
//    }
    for(uf <- ufos){
      uf.toList
    }
    val end = System.currentTimeMillis()
    println("time taken "+((end-start)/1000d))


//    val uf  = UnionFindPersistentMy(1 ,3 , 4)
//    val uf2 = uf.union(3, 4)
//    val uf3 = uf2.union(4, 1)
//    uf.findSetRep(4)
//    println(s"uf = $uf")
//    println(s"uf2 = $uf2")
//    println(s"uf3 = $uf3")
//    println(uf3)
//    println( uf.contains(4) )
//    println( uf.contains(2) )
//    println("Hello")
  }

}
