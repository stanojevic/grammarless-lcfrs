package edin.constituency.representation

import java.io.PrintWriter

import edin.algorithms.RichSeq._
import edin.algorithms.AutomaticResourceClosing.linesFromFile

import scala.annotation.tailrec

object PennFormatParser {

  def writeToFile(trees: Seq[ConstNode], fn:String, encoding:String = "utf-8") : Unit = {
    val fh = new PrintWriter(fn, encoding)
    for((tree, i) <- trees.zipWithIndex){
      fh.println(toString(tree))
    }
    fh.close()
  }

  def toString(tree:ConstNode) : String = tree match {
    case ConstNode(label, Nil) =>
      s"($label ${tree.word})"
    case ConstNode(label, children) =>
      s"($label ${children map toString mkString " "})"
  }

  // warning: this file handle will be open until stream is read until the end
  def fromFile(fn:String) : Stream[ConstNode] =
    linesFromFile(fn).
      toStream.
      filterNot(_ matches "^ *$").
      splitByContentAsBeginning(_ startsWith "(").
      map(_ mkString " ").
      map(fromString)

  def fromString(s:String) : ConstNode = {
    val tokens = s.replaceAll("""\(|\)""", " $0 ").split(" +").toList.filterNot(_=="")
    val node = recLL0(List(), tokens)
    assignSpans(node)
    node
  }

  @tailrec
  private def recLL0(stack:List[Either[String, ConstNode]], buffer:List[String]) : ConstNode = buffer match {
    case Nil =>
      stack match {
        case List(Right(node)) => node
        case _                 => sys.error("failed parsing")
      }
    case ")"::btail =>
      val suffix  = stack.dropWhile(_!=Left("(")).tail
      val newNode = stack.takeWhile(_!=Left("(")).reverse match {
        case Right(node)::Nil =>
          assert(btail.isEmpty)
          node
        case Left(label) :: Left(word) :: Nil =>
          val node = ConstNode(label=label, children=Nil)
          node.setTerminalInfo(word, -9999)
          node
        case Left(label) :: children =>
          ConstNode(label=label, children = children.map(_.right.get))
        case _ =>
          sys.error("you should not end up here")
      }
      recLL0(Right(newNode)::suffix, btail)
    case bhead::btail =>
      recLL0(Left(bhead)::stack, btail)
  }

  def assignSpans(node:ConstNode) : Unit =
    for((node, i) <- node.leafsSorted.zipWithIndex)
      node.setTerminalInfo(node.word, i)

}

