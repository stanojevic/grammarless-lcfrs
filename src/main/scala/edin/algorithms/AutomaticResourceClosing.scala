package edin.algorithms

import java.io.File

import scala.io.Source
import scala.util.{Failure, Success, Try}

object AutomaticResourceClosing {

  def linesFromFile(fn:String) : Iterator[String] = linesFromFile(fn, "UTF-8")
  def linesFromFile(fn:String, encoding:String) : Iterator[String] = linesFromFile(new File(fn), encoding)

  def linesFromFile(fn:File) : Iterator[String] = linesFromFile(fn, "UTF-8")

  def linesFromFile(fn:File, encoding:String) : Iterator[String] = new Iterator[String] {

    private var fh = Source.fromFile(fn, encoding)
    private val lines = fh.getLines()

    override def hasNext: Boolean = fh!=null && lines.hasNext

    override def next(): String =
      if (hasNext)
        Try(lines.next) match {
          case Success(s) =>
            if(!hasNext){
              fh.close()
              fh = null
            }
            s
          case Failure(e) =>
            throw e
        }
      else
        sys.error("you are trying to read a reasource that is already closed")

  }

}
