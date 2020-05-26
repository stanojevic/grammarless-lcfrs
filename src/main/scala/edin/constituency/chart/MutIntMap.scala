package edin.constituency.chart

import scala.reflect.ClassTag

/**
  * MutIntMap avoids boxing (as much as it can) and generalizes over different containers
  * It also doesn't allow removal of elements; that guarantees performance in case one
  * uses containers that rely on Dynamic Arrays
  */
object MutIntMap{

  def construct[V >: Null <: AnyRef](ttype:String, dim:Int)(implicit m:ClassTag[V]) : MutIntMap[V] = ttype match {
    case "array"       => new MutIntMapArray[V](dim)
    case "bit-simple"  => new MutIntMapBitSimple[V]()
    case "bit-dynamic" => new MutIntMapBitDynamic[V]()
    case "tree-map"    => new MutIntMapScalaMap[V](true)
    case "hash-map"    => new MutIntMapScalaMap[V](false)
  }

}

// trait MutIntMap[V] extends collection.mutable.Map[Int, V]{
trait MutIntMap[V] extends collection.mutable.Map[Int, V] {

  override final def iterator: Iterator[(Int, V)] = toStream.iterator

  @inline
  def primitivePut(k:Int, v:V) : Unit

  @inline def delete(k:Int) : Unit

  @inline override  def isEmpty  : Boolean = size==0

  @inline override  def nonEmpty : Boolean = !isEmpty

  @inline def toStream: Stream[(Int, V)]

  override def toString: String = "IntMap("+toStream.map{case (k, v) => s"$k -> $v"}.mkString("\n")+")"

  override final def +=(kv: (Int, V)): this.type = {
    primitivePut(kv._1, kv._2)
    this
  }

  override final def -=(key: Int): this.type = {
    delete(key)
    this
  }

}

private class MutIntMapScalaMap[V](redblack:Boolean) extends MutIntMap[V] {

  private val storage =
    if(redblack) scala.collection.mutable.TreeMap[Int, V]()
    else         scala.collection.mutable.HashMap[Int, V]()

  private var ssize = 0

  override def size : Int = ssize

  override def primitivePut(k: Int, v: V): Unit = {
    if(! storage.contains(k)){
      ssize += 1
    }
    storage(k) = v
  }

  override def get(k: Int): Option[V] =
    storage.get(k)

  override def toStream: Stream[(Int, V)] =
    storage.toStream

  override def delete(k: Int): Unit =
    if(contains(k)){
      ssize -= 1
      storage.remove(k)
    }

}

private class MutIntMapBitDynamic[V]() extends MutIntMap[V] {

  private var bitmap = BitSetC()
  private val elems = scala.collection.mutable.ArrayBuffer[V]()

  override def contains(key: Int) : Boolean = bitmap contains key

  override def primitivePut(k: Int, v: V): Unit =
    if(bitmap contains k){
      elems(bitmap count_elems_less_than k) = v
    }else{
      bitmap = bitmap + k
      val pos = bitmap count_elems_less_than k

      elems.append(v)
      var i = bitmap.size-1
      while(i != pos){
        elems(i-1) = elems(i)
        i -= 1
      }
    }

  override def get(k: Int): Option[V] = if(bitmap contains k) Some(elems(bitmap count_elems_less_than k)) else None

  override def toStream: Stream[(Int, V)] = (bitmap.toList zip elems).toStream

  override def values: Iterable[V] = elems

  override def delete(k: Int): Unit =
    if(bitmap.contains(k)){
      val pos = bitmap count_elems_less_than k
      bitmap = bitmap - k
      elems.remove(pos)
    }

}

private class MutIntMapBitSimple[V]()(implicit m:ClassTag[V]) extends MutIntMap[V] {

  private var bitmap = BitSetC()
  private var elems = Array[V]()

  override def contains(key: Int) : Boolean = bitmap contains key

  override def size : Int = elems.length

  override def primitivePut(k: Int, v: V): Unit =
    if(bitmap contains k){
      elems(bitmap count_elems_less_than k) = v
    }else{
      bitmap = bitmap + k
      val elems2 = new Array[V](elems.length+1)
      val pos = bitmap count_elems_less_than k

      System.arraycopy(elems, 0, elems2, 0, pos)
      elems2(pos) = v.asInstanceOf[V]
      System.arraycopy(elems, pos, elems2, pos+1, elems.length - pos)

      elems = elems2
    }

  override def get(k: Int): Option[V] = if(bitmap contains k) Some(elems(bitmap count_elems_less_than k)) else None

  override def toStream: Stream[(Int, V)] = (bitmap.toList zip elems).toStream

  override def values: Iterable[V] = elems

  override def delete(k: Int): Unit =
    if(bitmap.contains(k)){
      val pos = bitmap count_elems_less_than k
      bitmap = bitmap - k
      val elemsOld = elems
      elems = new Array[V](elems.length-1)
      System.arraycopy(elemsOld, 0, elems, 0, pos-1)
      System.arraycopy(elemsOld, pos+1, elems, pos, elemsOld.length-pos)
    }

}

private class MutIntMapArray[V >: Null <: AnyRef](maxSize:Int) extends MutIntMap[V] {

  private val elems = Array.ofDim[AnyRef](maxSize)

  private var ssize = 0

  override def size : Int = ssize

  override def primitivePut(k: Int, v: V): Unit = {
    if(elems(k) == null){
      ssize += 1
    }
    elems(k) = v
  }

  override def get(k: Int): Option[V] = Option(elems(k).asInstanceOf[V])

  override def toStream: Stream[(Int, V)] =
    for( i <- elems.indices.toStream if elems(i) != null )
      yield (i, elems(i).asInstanceOf[V])

  override def values: Stream[V] =
    elems.toStream.filterNot(_==null).asInstanceOf[Stream[V]]

  override def delete(k: Int): Unit =
    if(elems(k) != null){
      ssize -= 1
      elems(k) = null
    }

}
