package edin.constituency

import edin.constituency.agenda.AgendaPriority
import edin.constituency.chart.Query.{EqCond, Star}
import edin.constituency.chart.{ChartList, ChartTable, ChartTrie}
import edin.constituency.grammar.Item
import edin.constituency.representation.PennFormatParser

object Main {

  def main4(args : Array[String]) : Unit = {
    val penn_mrg = "/home/milos/Projects/CCG-translator/data/penn_treebank/wsj/00/wsj_0002.mrg"
    PennFormatParser.fromFile(penn_mrg).foreach{ tree =>
      println(tree)
      tree.deleteEmptyNodes.binarized.collapseUnaries.addDummyRootNode.visualize("4")
    }
  }

  private case class MyItem(i:Int, j:Int, nt:Int, score:Double) extends Item[MyItem] {
    override var bestScore: Double = score
    override var priority: Double = score
    override def signature: List[Int] = List(i, j, nt)
  }

  def main(args:Array[String]) : Unit = {

    val i1 = MyItem(0, 1, 32, 2.1)
    val i2 = MyItem(0, 1, 32, 2.43)
    val i3 = MyItem(0, 1, 0, 2.42)

//     val agenda = new AgendaList[MyItem]()
    // val agenda = new AgendaPriority[MyItem](new ChartTrie(List(("hash-map", 2), ("hash-map", 2), ("hash-map", 32))), "Fibonacci")
    val agenda = new AgendaPriority[MyItem](new ChartList(), "Fibonacci")
//    val agenda = new AgendaUnique[MyItem](new ChartList())
    println(agenda)
    agenda.add(i1)
    println(agenda)
    agenda.add(i3)
    println(agenda)
    agenda.add(i2)
    println(agenda)

    println("popping")
    while(agenda.nonEmpty){
      println("pop "+agenda.pop())
      println(agenda)
    }

    println("Hello")
  }

  def main2(args:Array[String]) : Unit = {

    val c = if(false){
      new ChartTable[String](List(2,3,1,5), true)
    }else if(true){
      new ChartTrie[String](List(
        ("bit-dynamic", 2),
        ("bit-dynamic", 3),
        ("bit-dynamic", 1),
        ("bit-dynamic", 5),
      ))
    }else{
      new ChartList[String]()
    }

    c += List(0, 0, 0, 0) -> "nulti"
    c += List(0, 1, 0, 0) -> "prvi"
    c -= List(0, 1, 0, 0)

    for(i <- c.query(List(Star, EqCond(1), EqCond(0), EqCond(0)))){
      println("nalaz: "+i)
    }

    println(c)

  }


}

