package edin.constituency.representation

import java.io.PrintWriter

import edin.algorithms.RichSeq._
import edin.algorithms.AutomaticResourceClosing.linesFromFile

object ExportFormatParser {

  def writeToFile(trees: Seq[ConstNode], fn:String, encoding:String = "utf-8") : Unit = {
    val fh = new PrintWriter(fn, encoding)
    for((tree, i) <- trees.zipWithIndex){
      fh.println(toString(tree, i))
    }
    fh.close()
  }

  def fromFile(fn:String, encoding:String = "utf-8") : Stream[ConstNode] =
    linesFromFile(fn, encoding).
      toStream.
      dropWhile(! _.startsWith("#BOS")).
      splitByContentAsBeginning(_ startsWith "#BOS").
      map{entries =>
        val children = collection.mutable.Map[Int, List[ConstNode]]().withDefault(Nil)
        var position = 0

        for{
          line   <- entries.tail.init
          fields =  line split "\t+"
        }{
          assert(fields.length==5 || fields.length==6) // this is export after treetools transformation ; in the "original" version it has 6 fields
          val labelField = if(fields.length==5) 1 else 2
          if(fields.head.startsWith("#5") || fields.head.startsWith("#6")){
            // it's a phrase
            val nodeId = fields(0).tail.toInt
            val nt = fields(labelField)
            val node = ConstNode(nt, children(nodeId).sortBy(_.indices.min))
            val parent = fields(fields.length-1).toInt
            children(parent) = node :: children.getOrElse(parent, Nil)
          }else{
            // it's a word
            val word = fields(0)
            val tag  = fields(labelField)
            val parent = fields(fields.length-1).toInt
            val node = ConstNode(tag, Nil)
            node.setTerminalInfo(word, position)
            position += 1
            children(parent) = node :: children.getOrElse(parent, Nil)
          }

        }

        ConstNode("ROOT", children(0).sortBy(_.indices.min))
      }

  // deletes self repetitive unary nodes like NP->NP
  private def deleteDummyNull(node: ConstNode) : ConstNode = node.children match {
    case List(child) if node.label == child.label => deleteDummyNull(child)
    case children => node.copy(children = children map deleteDummyNull)
  }

  def toString(treeArg:ConstNode, sentId:Int=1) : String = {
    val header = s"#BOS $sentId"
    val footer = s"#EOS $sentId"
    if(treeArg.children.isEmpty){
      return s"$header\n${treeArg.word}\t${treeArg.label}\t--\t--\t0\n$footer"
    }
    val graph = scala.collection.mutable.Map[Int, Int]()
    val labels = scala.collection.mutable.Map[Int, String]()
    val nonTermCount = treeArg.allNodes.count(_.children.nonEmpty) - (if(treeArg.label=="ROOT") 1 else 0)
    val subTrees = treeArg.label match {
      case "ROOT" => treeArg.children
      case _ => List(treeArg)
    }
    var currFreeId = 500+nonTermCount-1
    def visit(node:ConstNode, currParent:Int) : Unit = {
      val nodeId = if(node.children.isEmpty){
        node.indices.head
      } else{
        currFreeId-=1
        currFreeId+1
      }
      graph(nodeId)  = currParent
      labels(nodeId) = node.label
      node.children.foreach(visit(_, nodeId))
    }
    subTrees.foreach(visit(_, 0))

    val termStr = treeArg.words.zipWithIndex.map{case (word, i) => s"$word\t${labels(i)}\t--\t--\t${graph(i)}"}.mkString("\n")
    val nonTermStr = (500 to 500+nonTermCount-1).map(i => s"#$i\t${labels(i)}\t--\t--\t${graph(i)}").mkString("\n")
    List(header, termStr, nonTermStr, footer).mkString("\n").replaceAllLiterally("\n\n", "\n")
  }

}
