package edin.constituency.chart

import edin.constituency.chart.Query.Query

class ChartOptimized[B <: AnyRef](
                                   ttype : String,
                                   dims  : List[(String, Int)],
                                   perms : List[List[Int]]
                                 ) extends Chart[B] {

  private val allPerms =
    if(perms.isEmpty)
      sys.error("if you want all permutations then you have to explicitly create them")
      // perms.indices.toList.permutations.toList
    else
      perms

  private val hashes = Chart.construct[Chart[B]]("table-multidim-array", dims.map(_ => ("", dims.size)))
  private val mainChart = hashes.values.head

  private def permute[A](query: List[A], perm:List[Int]) : List[A] = {
    val q = query.toIndexedSeq
    perm map q
  }

  override def newEmptyVersion[V <: AnyRef]: Chart[V] =
    new ChartOptimized[V](ttype, dims, perms)

  override def query(query: Query): Stream[B] = {
    val (q, perm) = query.zipWithIndex.sortBy(_.toString).unzip    /// TODO TODO TODO
    hashes(perm).query(q)
  }

  override protected val name: String = s"ChartOptimized+$ttype"

  override def +=(kv: (List[Int], B)): ChartOptimized.this.type = {
    for(perm <- allPerms){
      hashes(perm) += permute(kv._1, perm) -> kv._2
    }
    this
  }

  override def -=(key: List[Int]): ChartOptimized.this.type = {
    for(perm <- allPerms){
      hashes(perm) -= permute(key, perm)
    }
    this
  }

  override def iterator: Iterator[(List[Int], B)] =
    hashes.values.head.iterator

}
