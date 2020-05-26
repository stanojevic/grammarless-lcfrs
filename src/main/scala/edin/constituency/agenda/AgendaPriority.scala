package edin.constituency.agenda

import edin.constituency.chart.{Chart, ChartOptimized}
import edin.constituency.grammar.Item
import org.jheaps.AddressableHeap
import org.jheaps.array.{BinaryArrayAddressableHeap, DaryArrayAddressableHeap}
import org.jheaps.tree.{FibonacciHeap, LeftistHeap, PairingHeap}
import java.lang.{Double => DoubleBox}

import org.jheaps.AddressableHeap.Handle

import scala.reflect.ClassTag
import scala.collection.JavaConverters._

// here it matters if you have duplicates because each lookup is log(A) where A is the agenda size
class AgendaPriority[I <: Item[I]](handleMap: Chart[Option[Handle[DoubleBox, I]]], heapType:String)(implicit eClassTag: ClassTag[I]) extends Agenda[I] {

  require(!handleMap.isInstanceOf[ChartOptimized[_]], "chart-optimized is not the right choice")

  override def head : I = heap.findMin().getValue

  private var heap:AddressableHeap[DoubleBox, I] = heapType match {
    case "Fibonacci"                    => new              FibonacciHeap()
    case "Binary"   | "BinaryNoHeapify" => new BinaryArrayAddressableHeap()
    case "4ary"     | "4aryNoHeapify"   => new   DaryArrayAddressableHeap(4)
    case "8ary"     | "8aryNoHeapify"   => new   DaryArrayAddressableHeap(8)
    case "Pairing"                      => new                PairingHeap()
    case "Leftist"                      => new                LeftistHeap()
    case _                              => throw new Exception(s"heap $heapType unsupported")
  }

  def toStream : Stream[I] =
    for((_, Some(handle)) <- handleMap.toStream)
      yield handle.getValue

  private def getHandle(el: I) : Option[Handle[DoubleBox, I]] =
    for(Some(handle) <- handleMap.get(el.signature))
      yield handle

  private def storeHandle(handle: Handle[DoubleBox, I]) : Unit =
    handleMap(handle.getValue.signature) = Some(handle)

  private def markDeletedHandle(handle: Handle[DoubleBox, I]) : Unit =
    handleMap(handle.getValue.signature) = None

  def size : Int = heap.size().toInt

  private def contains(el:I) : Boolean =
    getHandle(el).nonEmpty

  private def getSameElement(el:I) : Option[I] =
    getHandle(el).map(_.getValue)

  private def insertSmart(el:I) : Unit =
    getHandle(el) match{
      case Some(handle) if el.priority > priorityTransform(handle.getKey) =>
        if(el.priority > priorityTransform(handle.getKey)){
          // delete+insert is better then decreaseKey because not only score but also backpointer might change
          // it's two timpes slower than decreaseKey but safer from potential future bugs
          handle.delete()
          handleMap(el.signature) = None
          insert(el)
        }
      case None =>
        insert(el)
    }

  private def increaseKey(el:I, newPriority:Double) : Unit =
    getHandle(el).get.decreaseKey(priorityTransform(newPriority))

  @inline
  private def priorityTransform(priority:Double) : Double = -priority

  @inline
  private def toBoxedArray : List[Double] => Array[DoubleBox] = _.map(Double.box).toArray

  private def insertChunk(els:List[I], priorities:List[Double]) : Unit = (size, heapType) match {
    case (0, "Binary") =>
      val h = BinaryArrayAddressableHeap.heapify(toBoxedArray(priorities map priorityTransform), els.toArray)
      heap = h
      for(handler <- h.handlesIterator().asScala)
        storeHandle(handler)
    case (0, "4ary") =>
      val h = DaryArrayAddressableHeap.heapify(4, toBoxedArray(priorities map priorityTransform), els.toArray)
      heap = h
      for(handler <- h.handlesIterator().asScala)
        storeHandle(handler)
    case (0, "8ary") =>
      val h = DaryArrayAddressableHeap.heapify(8, toBoxedArray(priorities map priorityTransform), els.toArray)
      heap = h
      for(handler <- h.handlesIterator().asScala)
        storeHandle(handler)
    case (_, _) =>
      for((el, priority) <- els zip priorities)
        add(el)
  }

  @inline
  private def insert(el:I) : Unit = {
    val handle = heap.insert(priorityTransform(el.priority), el)
    storeHandle(handle)
  }

  @inline
  override def add(i: I): Unit = insertSmart(i)

  override def addAxioms(axioms: Seq[I]): Unit = {
    require(isEmpty)
    insertChunk(axioms.toList, axioms.map(_.priority).toList)
  }

  def pop() : I = extractMax()._1

  private def extractMax() : (I, Double) = {
    val handle = heap.deleteMin()
    val el = handle.getValue
    val priority = priorityTransform(handle.getKey)
    markDeletedHandle(handle)
    (el, priority)
  }

}

