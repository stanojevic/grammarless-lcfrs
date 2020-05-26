package edin.algorithms

import spire.syntax.cfor.cforRange

class MathArray(val array:Array[Float]) extends Serializable {

  val length : Int = array.length

  @inline
  def apply(pos:Int) : Float = array(pos)

  @inline
  def update(pos:Int, v:Float) : Unit = array(pos) = v

  def +=(other:MathArray) : Unit = {
    val y = other.array
    assert(array.length == y.length)
    cforRange(0 until array.length){ i =>
      array(i) += y(i)
    }
  }

  def /=(y:Float) : Unit =
    cforRange(0 until array.length){ i =>
      array(i)/=y
    }

  def +(other:MathArray) : Array[Float] = {
    val y = other.array
    assert(array.length == y.length)
    val z = new Array[Float](array.length)
    cforRange(0 until array.length){ i =>
      z(i) = array(i)+y(i)
    }
    z
  }

  def map(f:Double => Double) : MathArray = {
    val xs = this.copy()
    val xsA = xs.array
    cforRange(0 until xs.length){ i =>
      xsA(i)=f(xsA(i).toDouble).toFloat
    }
    xs
  }

  def inplaceMap(f:Float => Float) : Unit =
    cforRange(0 until array.length){ i =>
      array(i)=f(array(i))
    }

  def copy() : MathArray = {
    val z = new Array[Float](array.length)
    cforRange(0 until array.length){ i =>
      z(i) = array(i)
    }
    new MathArray(z)
  }

  def toArray : Array[Float] = array.clone()

  def toList : List[Float] = array.toList

}

object MathArray{

  @inline
  def apply(array:Array[Float]): MathArray = new MathArray(array)

  @inline
  def apply(size:Int): MathArray = MathArray(Array.ofDim[Float](size))

}

