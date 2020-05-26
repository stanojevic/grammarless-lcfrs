package edin.general

import scala.collection.mutable.{Map => MutMap}
import org.mapdb.{DBMaker, Serializer}
import scala.collection.JavaConverters._

object StringMapDatabase{

  private val opennedMaps = MutMap[String, StringMapDatabase[_]]()

  def create[V<:AnyRef](fn:String, readOnly:Boolean) : StringMapDatabase[V] = {
    opennedMaps.get(fn) match {
      case Some(m) =>
        m.asInstanceOf[StringMapDatabase[V]]
      case None    =>
        val m = new StringMapDatabase[V](fn, readOnly)
        opennedMaps += fn -> m
        m
    }
  }

  def main(args:Array[String]) : Unit = {
    val mmap = new StringMapDatabase[Array[Float]]("something2.db", false)
    for(i <- scala.util.Random.shuffle((0 to 60000).toList)){
      val key = s"This is a $i sentence."
      if(true) {
        println(i+" "+mmap.get(key).get.toList)
      }else{
        println(i)
        mmap += key -> Array(0f, i.toFloat, 2f ,3f, 5.23f)
      }
    }
    println(mmap.get("dlkfjalsdkjfalsjdfla"))
    mmap.close()
    println("Hello")
  }

}

class StringMapDatabase[V<:AnyRef] private (fn:String, readOnly:Boolean) extends MutMap[String, V]{

//  private val initDbMaker = DBMaker.fileDB(fn).fileMmapEnableIfSupported().closeOnJvmShutdown()
  private val initDbMaker = DBMaker.fileDB(fn).closeOnJvmShutdown()
  private val db   = if(readOnly) initDbMaker.readOnly().make() else initDbMaker.make()
  private val mmap = db.hashMap("map", Serializer.STRING, db.getDefaultSerializer).createOrOpen()
  // private val mmap = db.treeMap("map", Serializer.STRING, db.getDefaultSerializer).createOrOpen()

  private var isOpenned = true
  def close() : Unit = if(isOpenned){
    isOpenned = false
    StringMapDatabase.opennedMaps.remove(fn)
    db.close()
  }
  override def finalize(): Unit = close()

  override def +=(kv: (String, V)): StringMapDatabase.this.type = {
    mmap.put(kv._1, kv._2)
    this
  }

  override def get(key: String): Option[V] =
    Option(mmap.get(key).asInstanceOf[V])

  override def -=(key: String): StringMapDatabase.this.type = {
    mmap.remove(key)
    this
  }

  override def iterator: Iterator[(String, V)] =
    mmap.entrySet().iterator().asScala.map(e => (e.getKey, e.getValue.asInstanceOf[V]))

}

