package edin.constituency.representation

import java.io.File
import edin.general.visualizers.TreeVisualizer.{Color, Shape, SimpleTreeNode}
import scala.util.Try

object ConstNode{

  def getFormat(fn:String) : String = fn.replaceAll(".*\\.", "")

  def saveTreesToFile(trees:Seq[ConstNode], fn:String) : Unit = getFormat(fn) match {
    case "export" => ExportFormatParser.writeToFile(trees, fn)
    case "mrg"    => PennFormatParser.  writeToFile(trees, fn)
    case format   => sys.error(s"unknown trees format $format for $fn")
  }
  def loadTreesFromFile(fn:String) : Stream[ConstNode] = getFormat(fn) match {
    case "export" => ExportFormatParser.fromFile(fn)
    case "mrg"    => PennFormatParser.  fromFile(fn)
    case format   => sys.error(s"unknown trees format $format for $fn")
  }

}

final case class ConstNode(label:String, children:List[ConstNode]){

  var headChildIndex    = -1
  var headTerminalIndex = -1

  var attribs = Set[String]()

  lazy val indices : scala.collection.immutable.BitSet = children match {
    case Nil => scala.collection.immutable.BitSet(terminalIndex)
    case _   => children.map(_.indices).reduce(_ union _)
  }

  lazy val spans : List[(Int, Int)] = children match {
    case Nil if label=="-NONE-" =>
      List((terminalIndex, terminalIndex))
    case Nil =>
      List((terminalIndex, terminalIndex+1))
    case _ => children.flatMap(_.spans).sortBy(_._1).map(List(_)).reduce{ (xs, ys) =>
      val x = xs.last
      val y = ys.head

      if(x._2 == y._1)
        (xs.init++((x._1, y._2)::ys.tail))
      else
        xs ++ ys
    }
  }

  def copy(label:String=label, children:List[ConstNode]=children) : ConstNode = {
    val n = ConstNode(label, children)
    n.attribs = this.attribs
    if(children.isEmpty)
      n.setTerminalInfo(this.terminalWord, this.terminalIndex)
    n
  }

  def words : List[String] = leafsSorted.map(_.word)

  def word : String = {
    assert(children.isEmpty)
    terminalWord
  }

  private var terminalIndex = -9999
  private var terminalWord:String = _
  def setTerminalInfo(word:String, index:Int) : Unit = {
    require(children.isEmpty)
    this.terminalIndex = index
    this.terminalWord = word
  }

  def leafsSorted : List[ConstNode] = leafs.sortBy(_.indices.headOption)

  def leafs : List[ConstNode] = children match {
    case Nil => List(this)
    case _ => children.flatMap(_.leafs)
  }

  def visualize(graphLabel:String="", fileType:String="pdf") : Unit =
    this.toSimpleTreeNode.visualize(graphLabel=graphLabel, fileType=fileType)

  def saveVisual(fn:String, graphLabel:String="", fileType:String="pdf") : Unit =
    this.toSimpleTreeNode.saveVisual(new File(s"$fn.$fileType"), graphLabel=graphLabel, fileType=fileType)

  private def toSimpleTreeNode : SimpleTreeNode = children match {
    case Nil =>
      val attribs_str = if(attribs.isEmpty) "" else "\n"+(attribs mkString " ")
      SimpleTreeNode(
        label    = s"$label\n$word$attribs_str\n${indices.head}",
        children = Nil,
        shape    = Shape.RECTANGLE,
        position = terminalIndex,
        color    = Color.LIGHT_BLUE)
    case children =>
      val attribs_str = if(attribs.isEmpty) "" else "\n"+(attribs mkString " ")
      val (shape, color) =
        if      ( label startsWith "BIN>_" ) (Shape.ELLIPSE, Color.LIGHT_BLUE)
        else if ( children.tail.isEmpty    ) (Shape.HEXAGON, Color.GREEN     )
        else                                 (Shape.ELLIPSE, Color.RED       )
      SimpleTreeNode(
        label    = label+attribs_str,
        children = children.map(_.toSimpleTreeNode),
        shape    = shape,
        color    = color)
  }

  override def toString: String = children match {
    case Nil      => s"($label $word)"
    case children => s"($label "+children.mkString(" ")+")"
  }

  def allNodes:List[ConstNode] = allNodesPreorder

  def allNodesPreorder:List[ConstNode] = children match {
    case Nil => List(this)
    case children => this :: children.flatMap(_.allNodesPreorder)
  }

  def allNodesPostorder:List[ConstNode] = children match {
    case Nil => List(this)
    case children => children.flatMap(_.allNodesPostorder) :+ this
  }

  def addDummyRootNode : ConstNode = this.label match {
    case "ROOT" => this
    case _ => ConstNode("ROOT", List(this))
  }
  def deleteDummyRootNode : ConstNode = this match {
    case ConstNode("ROOT", List(child)) => child
    case _ => this
  }

  def deleteEmptyNodes : ConstNode = {
    val newNode = deleteEmptyNodesRec(this).get
    PennFormatParser.assignSpans(newNode)
    newNode
  }

  private def deleteEmptyNodesRec(node: ConstNode) : Option[ConstNode] = node.children match {
    case Nil if node.label == "-NONE-" => None
    case Nil => Some(node.copy())
    case children =>
      node.children.flatMap(deleteEmptyNodesRec) match {
        case Nil =>
          None
        case newChildren =>
          val newNode = node.copy(children = newChildren)
          newNode.attribs = newNode.attribs.filterNot(x => Try(Integer.parseInt(x)).isSuccess)
          Some(newNode)
      }
  }

  def binarizedLossy : ConstNode = children match {
    case List() | List(_) | List(_, _)  => this.copy(children = children.map(_.binarizedLossy))
    case leftmost::rest => this.copy(children = List(leftmost, mergeForBinarizationLossy(rest)).map(_.binarizedLossy))
  }
  private def mergeForBinarizationLossy(xs:List[ConstNode]) : ConstNode =  xs.reduceRight((a, b) => ConstNode("∅", List(a, b)))

  def binarized : ConstNode = children match {
    case List() | List(_) | List(_, _)  => this.copy(children = children.map(_.binarized))
    case leftmost::rest => this.copy(children = List(leftmost, mergeForBinarization(rest)).map(_.binarized))
  }
  private def mergeForBinarization(xs:List[ConstNode]) : ConstNode = ConstNode(label = "BIN>:"+xs.map(_.label).mkString(":"), children = xs)

  def deBinarized : ConstNode = children match {
    case List(left, right) if left.label == "∅" && right.label == "∅" =>
      this.copy(children = left.deBinarized.children ++ right.deBinarized.children)
    case List(left, right) if right.label.startsWith("BIN>:") || right.label == "∅" =>
      this.copy(children = left.deBinarized :: right.deBinarized.children)
    case List(left, right) if left.label == "∅" =>
      this.copy(children = left.deBinarized.children :+ right.deBinarized)
    case children =>
      this.copy(children = children.map(_.deBinarized))
  }

  def deCollapseUnaries : ConstNode = children match {
    case List(child) =>
      val newChild = child.deCollapseUnaries
      val labels = this.label.split("\\+").toList
      val newSelf1 = labels.foldRight(child){case (label, child) => ConstNode(label = label, children = List(child))}
      this.copy(label = newSelf1.label, children=newSelf1.children)
    case children =>
      this.copy(children = children.map(_.deCollapseUnaries))
  }

  def collapseUnaries : ConstNode = children match {
    case List(child) =>
      val newChild = child.collapseUnaries
      newChild.children match {
        case List(subChild) =>
          this.copy(
            label = label+"+"+newChild.label,
            children = List(subChild)
          )
        case _ =>
          this.copy(children = List(newChild))
      }
    case children =>
      this.copy(children = children.map(_.collapseUnaries))
  }

}

