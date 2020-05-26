package edin.constituency.chart
import edin.constituency.chart.Query.{EqCond, Pred, Query, Star}


class ChartTrie[B <: AnyRef](dims:List[(String, Int)]) extends Chart[B] {

  override protected val name: String = "ChartTrie "+dims.map(_._1).mkString(" ")

  private val fullTrie: MutIntMap[AnyRef] = MutIntMap.construct[AnyRef](dims.head._1, dims.head._2)

  private def lookup(query: Query, trie:AnyRef) : Stream[B] = query match {
    case Nil =>
      Stream(trie.asInstanceOf[B])
    case Star :: qs =>
      trie.asInstanceOf[MutIntMap[AnyRef]].values.flatMap(lookup(qs, _)).toStream
    case EqCond(i) :: qs =>
      trie.asInstanceOf[MutIntMap[AnyRef]].get(i).toStream.flatMap(lookup(qs, _))
    case Pred(p) :: qs =>
      trie.asInstanceOf[MutIntMap[AnyRef]].toStream.filter(x => p(x._1)).flatMap(x => lookup(qs, x._2))
  }

  private def storeEntry(address: List[Int], dims:List[(String, Int)], trie:MutIntMap[AnyRef], b: B) : Unit = (address : @unchecked) match {
    case i::Nil =>
      trie.primitivePut(i, b)
    case i::is =>
      val subTrie = trie.get(i) match {
        case None =>
          val st = MutIntMap.construct[AnyRef](dims.head._1, dims.head._2)
          trie.primitivePut(i, st)
          st
        case Some(st) =>
          st.asInstanceOf[MutIntMap[AnyRef]]
      }
      storeEntry(is, dims.tail, subTrie, b)
  }

  override def query(query: Query): Stream[B] = {
    require(query.size == dims.size)
    lookup(query, fullTrie)
  }

  override def +=(kv: (List[Int], B)): ChartTrie.this.type = {
    require(kv._1.size == dims.size)
    storeEntry(kv._1, dims, fullTrie, kv._2)
    this
  }

  override def iterator: Iterator[(List[Int], B)] = iteratorRec(fullTrie).iterator

  private def iteratorRec(tree:AnyRef) : Stream[(List[Int], B)] = tree match {
    case tree:MutIntMap[_] =>
      for{
        (a, c:AnyRef) <- tree.toStream
        (as, d) <- iteratorRec(c)
      }
        yield (a::as, d)
    case d =>
      Stream((Nil, d.asInstanceOf[B]))
  }

  override def -=(key: List[Int]): ChartTrie.this.type = {
    deleteRec(key, fullTrie)
    this
  }

  private def deleteRec(key: List[Int], tree:MutIntMap[AnyRef]) : Unit = (key : @unchecked) match {
    case k :: Nil => tree.delete(k)
    case k :: ks  =>
      val subTree = tree(k).asInstanceOf[MutIntMap[AnyRef]]
      deleteRec(ks, subTree)
      if(subTree.isEmpty){
        tree.delete(k)
      }
  }

  override def newEmptyVersion[V <: AnyRef]: Chart[V] = new ChartTrie[V](dims)
}


