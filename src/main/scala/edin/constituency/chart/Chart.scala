package edin.constituency.chart

import edin.constituency.chart.Query.{EqCond, Query}

trait Chart[B] extends collection.mutable.Map[List[Int], B] {

  def newEmptyVersion[V <: AnyRef] : Chart[V]

  def query(query:Query) : Stream[B]

  protected val name : String

  override def remove(key: List[Int]): Option[B] = super.remove(key)

  override final def get(key: List[Int]): Option[B] = query(key.map(EqCond)).headOption

  override final def toString(): String =
    s"$name (\n"+
    iterator.map{case (k, v) => s"\t"+(k mkString " ")+" -> $v"}.mkString("\n")+
    "\n)"

}

object Chart{

  def construct[B <: AnyRef](ttype:String, dims:List[(String, Int)]) : Chart[B] = ttype match {
    case "list"                 => new ChartList()
    case "table-multidim-array" => new ChartTable(dims.map(_._2), true)
    case "table-single-hashmap" => new ChartTable(dims.map(_._2), false)
    case "trie"                 => new ChartTrie[B](dims)
  }

}
