package edin.nn.initialization

import edin.nn.model.TrainingController
import edu.cmu.dynet.{FloatVector, ParameterInit}
import org.apache.commons.math3.linear.{MatrixUtils, SingularValueDecomposition}
import spire.syntax.cfor.cforRange

import scala.util.Random
import scala.collection.mutable.{Map => MutMap}

object Saxe {

  private def randomVector(n:Int, r:Random) : Array[Double] = {
    val vec = new Array[Double](n)
    cforRange(0 until n){ i =>
      vec(i) = r.nextDouble()
    }
    vec
  }

  private def randomMatrix(n:Int, m:Int, r:Random) : Array[Array[Double]] = {
    val mat = new Array[Array[Double]](n)
    cforRange(0 until n){ i =>
      mat(i) = randomVector(m, r)
    }
    mat
  }

  private def matrixDoubleToMatrixFloat(ain:Array[Array[Double]]) : Array[Array[Float]] = {
    val aout = new Array[Array[Float]](ain.length)
    cforRange(0 until ain.length){ i =>
      aout(i) = new Array(ain(0).length)
      cforRange(0 until ain(0).length){ j =>
        aout(i)(j) = ain(i)(j).toFloat
      }
    }
    aout
  }

  private def newOrthogonalSquareMatrix(dim:Int, r:Random=null) : Array[Array[Float]] = {
    System.err.print(s"Computing orthogonal matrix ($dim, $dim) ... ")
    val startTime = System.currentTimeMillis()

    val ra = if(r==null) new Random else r
    val matrix = MatrixUtils.createRealMatrix(randomMatrix(dim, dim, ra))
    val Um = new SingularValueDecomposition(matrix).getU
    val Ud = Um.getData
    val Uf = matrixDoubleToMatrixFloat(Ud)

    System.err.println(s" took "+((System.currentTimeMillis()-startTime)/1000)+"s")
    Uf
  }

  private val cache = MutMap[Int, Array[Array[Float]]]()
  private def orthogonalSquareMatrix(dim:Int) : Array[Array[Float]] = {
    if(! cache.contains(dim)){
      cache(dim) = newOrthogonalSquareMatrix(dim)
    }
    cache(dim)
  }

  def saxeInit(dim:Int, activation:String) : ParameterInit =
    if(TrainingController.modelDefinitionDuringTraining){
      val gain = activation.toLowerCase match {
        case "relu" => math.sqrt(2).toFloat
        case _      => 1F
      }
      val ortho = orthogonalSquareMatrix(dim)
      var c = 0
      val res = new Array[Float](dim*dim)
      cforRange(0 until dim){ j=>
        cforRange(0 until dim){ i =>
          res(c) = ortho(i)(j)*gain
          c+=1
        }
      }
      ParameterInit.fromVector(new FloatVector(res))
    }else{
      ParameterInit.glorot(isLookup = false)
    }

//  def main(args:Array[String]) : Unit = {
//    import edin.nn.DyFunctions._
//    import edin.nn.DynetSetup
//    import edu.cmu.dynet.ParameterCollection
//    TrainingController.modelDefinitionDuringTraining=true
//    DynetSetup.init_dynet()
//    implicit val model = new ParameterCollection()
//
//    val x = addParameters((5, 5), initSaxe(5, "relu"))
//    System.err.println(s"dy[2,3]="+x.exp(2)(3).toFloat)
//    System.err.println(s"dy[3,2]="+x.exp(3)(2).toFloat)
//  }

}
