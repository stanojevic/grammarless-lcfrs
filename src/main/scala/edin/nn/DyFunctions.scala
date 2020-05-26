package edin.nn

import edu.cmu.dynet._
import edin.algorithms.BestKElements._
import edin.nn.initialization.Saxe
import edin.nn.layers.Dropout
import spire.syntax.cfor.cforRange

import scala.collection.BitSet
import scala.language.implicitConversions

object DyFunctions{

  def main(args:Array[String]) : Unit = {
    DynetSetup.init_dynet()

    {
      val shape = Dim(List(5, 5, 3), 2)
      val e  = Expression.randomUniform(shape, -1, 1)
      val e2 = vector(e.toSeq.toArray).reshape(shape)
      val a  = e.toBatchArray3d

      val b  = a.length
      val d1 = a(0).length
      val d2 = a(0)(0).length
      val d3 = a(0)(0)(0).length

      val e3 = batchTensor(a)

      println(e.pickBatchElem(1)(2)(2).toSeq mkString " ")
      println(e2.pickBatchElem(1)(2)(2).toSeq mkString " ")
      println(e3.pickBatchElem(1)(2)(2).toSeq mkString " ")
      println(a(1)(2)(2).toSeq mkString " ")



      System.exit(0)
    }




    val e = Expression.randomUniform(Dim(List(130, 40), 1), -1, 1)
    val a = e.toArray2d
    val b = vector(a.flatten).reshape(40, 130).T
    val c = matrix(a)
    println(e(120).toSeq mkString " ")
    println(a(120) mkString " ")
    println(b(120).toSeq mkString " ")
    println(c(120).toSeq mkString " ")
  }

  val Ɛ  : Float = 1e-7f  // very small constant to avoid division with zero
  val ∞  : Float =  1e9f  // very large number but not  Float.Inf
  val -∞ : Float = -1e9f  // very small number but not -Float.Inf

  def initAround(x:Float                    ) : ParameterInit = ParameterInit.uniform(x-Ɛ, x+Ɛ)
  def initConst( x:Float                    ) : ParameterInit = ParameterInit.const(x)
  def initSaxe(  d:Int, activation:String="") : ParameterInit = Saxe.saxeInit(d, activation)
  def initGlorot()                            : ParameterInit = ParameterInit.glorot(false)
  def initSaxeOrGlorot( n:Int, m:Int, activation:String="") : ParameterInit = if(n==m & n<1000) initSaxe(n, activation) else initGlorot()
  def init(name:String, n:Int, m:Int, activation:String="") : ParameterInit = name.toLowerCase match {
    case "saxe" =>
      assert(n==m, s"saxe initialization can be used only with square matrices; ($n, $m) isn't square")
      initSaxe(n, activation)
    case "saxe-or-glorot" =>
      initSaxeOrGlorot(n, m, activation)
    case "glorot" =>
      initGlorot()
    case ini =>
      sys.error(s"unknown initialization $ini")
  }

  // how to use checkpoints:
  // - https://dynet.readthedocs.io/en/latest/core.html#_CPPv4N5dynet16ComputationGraph10checkpointEv
  // - https://github.com/clab/dynet/blob/master/python/CHANGES.md#checkpoint--revert-mechanism-for-computation-graph
  @inline def checkpointStart()  : Unit = ComputationGraph.checkpoint()
  @inline def checkpointRevert() : Unit = ComputationGraph.revert()

  @inline def seqSeqInt2UnsignedVectorVector(x: Seq[Seq[Int]]) : UnsignedVectorVector = UnsignedVectorVector.Seq2UnsignedVectorVector(x map seqInt2UnsignedVector)
  @inline def seqInt2UnsignedVector(x: Seq[Int]) : UnsignedVector = UnsignedVector.Seq2UnsignedVector(x.map(_.toLong))

  @inline def selectRow (e:Expression, row:Int      ) : Expression = selectRows(e, List(row))
  @inline def selectCol (e:Expression, col:Int      ) : Expression = selectCols(e, List(col))
  @inline def selectRows(e:Expression, rows:Seq[Int]) : Expression = Expression.selectRows(e, seqInt2UnsignedVector(rows))
  @inline def selectCols(e:Expression, cols:Seq[Int]) : Expression = Expression.selectCols(e, seqInt2UnsignedVector(cols))

  @inline def argmaxWithScores(es:Seq[Float], k:Int):List[(Int, Float)] = es.zipWithIndex.bestKBy(k)(_._1).map(_.swap)
  @inline def argmaxWithScores(es:Expression, k:Int):List[(Int, Float)] = argmaxWithScores(es.toSeq, k)
  @inline def argmax(es:Seq[Float], k:Int):List[Int] = argmaxWithScores(es, k).map(_._1)
  @inline def argmax(es:Expression, k:Int):List[Int] = argmax(es.toSeq, k)
  @inline def argmax(es:Seq[Float]):Int = es.zipWithIndex.maxBy(_._1)._2 // argmax(es, 1).head
  @inline def argmax(es:Expression):Int = argmax(es.toSeq) // Expression.argmax(es).toSeq.count(_==0)

  @inline def argmaxBetaWithScores(es:Expression, beta:Float):List[(Int, Float)] = argmaxBetaWithScores(es.toSeq, beta)
  @inline def argmaxBetaWithScores(es:Seq[Float], beta:Float):List[(Int, Float)] = {
    val bestLogProb:Float = es.max
    val treshold = bestLogProb + math.log(1-beta)
    es.zipWithIndex.filter(_._1>=treshold).sortBy(-_._1).toList.map(_.swap)
  }

  @inline def sumBatches             : Expression               => Expression    = Expression.sumBatches
  @inline def avgBatches(e:Expression) : Expression                              = sumBatches(e)/e.batchSize
  @inline def zeros(dim: Dim)        :                             Expression    = Expression.zeros(dim, DynetSetup.currentDevice) // this different form is needed because of the default device argument
  @inline def ones( dim: Dim)        :                             Expression    = Expression.ones( dim, DynetSetup.currentDevice) // this different form is needed because of the default device argument
  @inline def dotProduct             : (Expression, Expression) => Expression    = Expression.dotProduct
  @inline def cmult                  : (Expression, Expression) => Expression    = Expression.cmult
  @inline def cdiv                   : (Expression, Expression) => Expression    = Expression.cdiv
  @inline def exp                    : Expression               => Expression    = Expression.exp
  @inline def log                    : Expression               => Expression    = Expression.log
  @inline def sumElems               : Expression               => Expression    = Expression.sumElems
  @inline def tanh                   : Expression               => Expression    = Expression.tanh
  @inline def relu                   : Expression               => Expression    = Expression.rectify
  @inline def sigmoid                : Expression               => Expression    = Expression.logistic
  @inline def logistic               : Expression               => Expression    = Expression.logistic
  @inline def transpose              : Expression               => Expression    = Expression.transpose
  @inline def softmax(e:Expression, d:Int) : Expression                          = Expression.softmax(e, d)
  @inline def softmax                : Expression               => Expression    = Expression.softmax(_, 0)
  @inline def logSoftmax             : Expression               => Expression    = Expression.logSoftmax
  @inline def scalar(x:Double)       : Expression                                = Expression.input(x.toFloat, DynetSetup.currentDevice)
  @inline def vector(x:Array[Float]) : Expression = {
    val e = new FloatVector(x)
    val exp = Expression.input(x.length, e, DynetSetup.currentDevice)
    DynetSetup.safeReference(x)
    DynetSetup.safeReference(exp)
    exp
  }
  @inline def matrix(x:Array[Array[Float]]) : Expression = vector(x.flatten).reshape(x(0).length, x.length).T
  @inline def batchTensor(a:Array[Array[Array[Array[Float]]]]) : Expression = {
    val bs = a.length
    val d1 = a(0).length
    val d2 = a(0)(0).length
    val d3 = a(0)(0)(0).length

    val v = new Array[Float](d1*d2*d3*bs)
    var c = 0
    cforRange(0 until bs) { b =>
      cforRange(0 until d3){ k =>
        cforRange(0 until d2){ j =>
          cforRange(0 until d1){ i =>
            v(c) = a(b)(i)(j)(k)
            c += 1
          }
        }
      }
    }
    vector(v).reshape((d1, d2, d3), bs)
  }
  @inline def tensor(a:Array[Array[Array[Float]]]) : Expression = {
    val d1 = a.length
    val d2 = a(0).length
    val d3 = a(0)(0).length

    val v = new Array[Float](d1*d2*d3)
    var c = 0
    cforRange(0 until d3){ k =>
      cforRange(0 until d2){ j =>
        cforRange(0 until d1){ i =>
          v(c) = a(i)(j)(k)
          c += 1
        }
      }
    }
    vector(v).reshape((d1, d2, d3))
  }

  @inline def batchExprs( exps : Seq[Expression]   ) : Expression = Expression.concatenateToBatch(exps)
  @inline def batchScalar(xs   : Array[Float]      ) : Expression = vector(xs).reshape(Dim(List(1), xs.length))
  @inline def batchScalar(xs   : Seq[Float]        ) : Expression = batchScalar(xs.toArray)
  // @inline def batchScalar_old(xs   : Seq[Float]        ) : Expression = batchVector(xs.map(Array(_)))
  @inline def batchVector(vecs : Seq[Array[Float]] ) : Expression = vector(vecs.flatten.toArray).reshape(Dim(List(vecs.head.length), vecs.length))
  // @inline def batchVector_old(vecs : Seq[Array[Float]] ) : Expression = batchExprs(vecs map vector)
  // Expression.concatenateToBatch(ExpressionVector.Seq2ExpressionVector(vecs map vector))
  // vector(vecs.flatten.toArray).reshape(d=(vecs(0).length, 1), b=vecs.size)

  @inline def batchZeroOutVector(inputBatch:Expression, batchIdsToModify:Seq[Int]) : Expression =
    if(batchIdsToModify.isEmpty){
      inputBatch
    }else{
      val batchSize = inputBatch.batchSize
      val mask = batchScalar(invertOneHot(manyHotArr(batchIdsToModify, batchSize)))
      inputBatch * mask
    }

  @inline def parameter(x:Parameter) : Expression = {
    var exp = Expression.parameter(x)
    if(DeviceManager.getDefaultDevice.getDevice_id != DynetSetup.currentDevice.getDevice_id)
      exp = exp.toDevice(DynetSetup.currentDevice)
    exp
  }

  @inline def manyHotArr(hots:BitSet, size:Int) : Array[Float] = {
    val a = Array.ofDim[Float](size)
    for(i <- 0 until size){
      if(hots contains i)
        a(i) = 1f
      else
        a(i) = 0f
    }
    a
  }

  @inline def manyHotArr(hots:Seq[Int], size:Int) : Array[Float] = {
    val a = Array.ofDim[Float](size)
    for(hot <- hots)
      a(hot) = 1f
    a
  }
  @inline def oneHotArr( hot:Int, size:Int ) : Array[Float] = manyHotArr(List(hot), size)
  @inline def oneHot(    hot:Int, size:Int ) : Expression   = vector(oneHotArr(hot, size))
  @inline def invertOneHot( a:Array[Float] ) : Array[Float] = a.map(x => if(x==1f) 0f else 1f)
  @inline def invertOneHot( e:Expression   ) : Expression   = 1-e // Expression.abs(e - 1)

  @inline def concatCols          (es:Expression*       ) : Expression   = Expression.concatenateCols(es)
  @inline def concat              (es:Expression*       ) : Expression   = Expression.concatenate(es)
  @inline def concatWithNull      (es:Expression*       ) : Expression   = Expression.concatenate(es.filter(_!=null))
  @inline def concatSeq           (es:Seq[Expression]   ) : Expression   = Expression.concatenate(es)
  @inline def concatSeqWithNull   (es:Seq[Expression]   ) : Expression   = Expression.concatenate(es.filter(_!=null))
  @inline def concatByDim         (es:Seq[Expression], d:Int) : Expression   = Expression.concatenate(es.filter(_!=null), d=d)
  @inline def logSumExp           (es:Seq[Expression]   ) : Expression   = if(es.size == 1) es.head else Expression.logSumExp(es)
  @inline def emax                (es:Seq[Expression]   ) : Expression   = Expression.max(es)
  @inline def esum                (es:Seq[Expression]   ) : Expression   = Expression.sum(es)
  @inline def eavg                (es:Seq[Expression]   ) : Expression   = Expression.average(es)
  @inline def averageLogSoftmaxes (es:Seq[Expression]   ) : Expression   = logSumExp(es) - log(es.size)
  @inline def pow                 (e:Expression, p:Float) : Expression   = Expression.pow(e, p)

  private var allDropoutEnabled             =  false
  @inline def dropoutIsEnabled    : Boolean =  allDropoutEnabled
  @inline def disableAllDropout() : Unit    = {allDropoutEnabled = false}
  @inline def enableAllDropout()  : Unit    = {allDropoutEnabled = true }
  @inline def dropout(x:Expression, d:Float) : Expression =
    if(dropoutIsEnabled)
      Expression.dropout(x, d)
    else
      x

  implicit class RichExpSeq(xs:Seq[Expression]){
    def econcat     : Expression = DyFunctions.concat(xs:_*)
    def econcatCols : Expression = DyFunctions.concatCols(xs:_*)
    def esum        : Expression = DyFunctions.esum(xs)
    def eavg        : Expression = DyFunctions.eavg(xs)
    def esumOrElse(x : => Expression) : Expression = if(xs.isEmpty) x else DyFunctions.esum(xs)
    def eavgOrElse(x : => Expression) : Expression = if(xs.isEmpty) x else DyFunctions.eavg(xs)
  }

  @inline implicit def double2expr(x:Double) : Expression = scalar(x)
  @inline implicit def int2expr(   x:Int   ) : Expression = scalar(x)
  @inline implicit def float2expr( x:Float ) : Expression = scalar(x)

  @inline implicit def int2Dim(x:Int) : Dim = Dim(x)
  @inline implicit def tupleTwo2Dim(x:(Int, Int)) : Dim = Dim(x._1, x._2)
  @inline implicit def tupleThree2Dim(x:(Int, Int, Int)) : Dim = Dim(x._1, x._2, x._3)
  @inline implicit def tupleTwoB2Dim(x:((Int, Int), Int)) : Dim = Dim(List(x._1._1, x._1._2), x._2)
  @inline implicit def tupleThreeB2Dim(x:((Int, Int, Int), Int)) : Dim = Dim(List(x._1._1, x._1._2, x._1._3), x._2)

  def trainerFactory(typ:String, lr:Float, clipping: Boolean)(implicit model:ParameterCollection) : Trainer = {
    val trainer = typ match{
      case "Adam" => new AdamTrainer(model, learningRate = lr)
      case "SGD"  => new SimpleSGDTrainer(model, learningRate = lr)
    }
    if(clipping)
      trainer.clipGradients()
    trainer
  }

  def activationFactory(activationName:String) : Expression => Expression =
    activationName.toLowerCase match {
      case "tanh"        => Expression.tanh
      case "sigmoid"     => Expression.logistic
      case "logistic"    => Expression.logistic
      case "relu"        => Expression.rectify
      case "linear"      => identity
      case "nothing"     => identity
      case "softmax"     => Expression.softmax(_, 0)
      case "logsoftmax"  => Expression.logSoftmax
      case "log_softmax" => Expression.logSoftmax
      case "log-softmax" => Expression.logSoftmax
    }

  def addParameters(d: Dim                                         )(implicit model:ParameterCollection) : SingularParameter = addParameters(d, dropConnect=0f)
  def addParameters(d: Dim, init: ParameterInit                    )(implicit model:ParameterCollection) : SingularParameter = addParameters(d, dropConnect=0f, init=init)
  def addParameters(d: Dim                     , dropConnect: Float)(implicit model:ParameterCollection) : SingularParameter = addParameters(d, dropConnect=dropConnect, init=initGlorot()) // new SingularParameter(model.addParameters(d), dropConnect)
  def addParameters(d: Dim, init: ParameterInit, dropConnect: Float)(implicit model:ParameterCollection) : SingularParameter = new SingularParameter(model.addParameters(d, init ), dropConnect)
  implicit def singular2exp      (singular : SingularParameter        ) : Expression         = singular.exp
  implicit def singularOpt2expOpt(singular : Option[SingularParameter]) : Option[Expression] = singular.map(_.exp)
  implicit def singularOpt2expOpt(singular : List[SingularParameter]  ) : List[Expression]   = singular.map(_.exp)

  class SingularParameter private[nn] (param:Parameter, dropConnect:Float){
    private var latestCG    : Int        = -1
    private var activeVal   : Expression = _
    private var currDevId   : Int     = _
    private val drop = Dropout(dropConnect)
    def exp : Expression = {
      if(latestCG != DynetSetup.cg_id){
        activeVal = drop(parameter(param))
        currDevId = DynetSetup.currentDevice.getDevice_id
        latestCG  = DynetSetup.cg_id
      }else if(currDevId != DynetSetup.currentDevice.getDevice_id){
        activeVal = activeVal.toDevice(DynetSetup.currentDevice)
      }
      activeVal
    }
  }

  implicit class RichLookupParameter(x:LookupParameter){
    def lookup(i:Int)             : Expression = Expression.lookup(x, i).toDevice(DynetSetup.currentDevice)
    def lookup(is:List[Int])      : Expression = Expression.lookup(x, seqInt2UnsignedVector(is)).toDevice(DynetSetup.currentDevice)
    def constLookup(i:Int)        : Expression = Expression.constLookup(x, i).toDevice(DynetSetup.currentDevice)
    def constLookup(is:List[Int]) : Expression = Expression.constLookup(x, seqInt2UnsignedVector(is)).toDevice(DynetSetup.currentDevice)
    def apply(i:Int):Expression = lookup(i)
    def apply(is:List[Int]):Expression = lookup(is)
  }

  implicit class RichExpression(x:Expression) {

    def toDevice(device:edu.cmu.dynet.internal.Device) : Expression = {
      if(DynetSetup.availableGPUs.isEmpty)
        x // it has to be like this because of the bug in DyNet
      else
        Expression.toDevice(x, device)
    }

    def toDevice(devName:String) : Expression = toDevice(DeviceManager.getGlobalDevice(devName))

    def rows : Int = x.dim().rows().toInt
    def cols : Int = x.dim().cols().toInt

    def splitCols : List[Expression] = (0 until cols).map(col).toList
    def splitRows : List[Expression] = (0 until rows).map(row).toList

    def zeros : Expression = DyFunctions.zeros(x.dim()) // x * 0
    def ones  : Expression = DyFunctions.ones( x.dim()) // x * 0 + 1

    def concat     (y: Expression) : Expression = DyFunctions.concat(x, y)
    def concatCols (y: Expression) : Expression = DyFunctions.concatCols(x, y)

    def ⊙     (y:Expression) : Expression = Expression.cmult(x, y)
    def cmult (y:Expression) : Expression = Expression.cmult(x, y)
    def pow   (y:Expression) : Expression = Expression.pow  (x, y)

    def sumElems   : Expression = Expression.sumElems(x)
    def meanElems  : Expression = Expression.meanElems(x)
    def stdElems   : Expression = Expression.stdElems(x)
    def sumBatches : Expression = Expression.sumBatches(x)
    def T          : Expression = Expression.transpose(x)

    def reshape(d: Dim): Expression = Expression.reshape(x, d)

    def toStr    : String       = this.str(true)
    def toFloat  : Float        = x.value().toFloat
    def toDouble : Double       = x.value().toFloat.toDouble
    def toSeq    : Seq[Float]   = x.value().toSeq
    def toArray  : Array[Float] = this.toSeq.toArray

    def toArray3d : Array[Array[Array[Float]]] = {
      assert(batchSize==1)
      toBatchArray3d(0)
    }
    def toArray2d : Array[Array[Float]] = {
      assert(batchSize==1)
      toBatchArray2d(0)
    }
    def toArray1d : Array[Float] = {
      assert(batchSize==1)
      toBatchArray1d(0)
    }

    def toBatchArray3d : Array[Array[Array[Array[Float]]]] = {
      val d = x.dim()
      val be = d.batchElems().toInt
      val ao = new Array[Array[Array[Array[Float]]]](be)
      val ai = toArray

      val List(ie, je, ke) = dims.take(3).map(_.toInt)
      cforRange(0 until be) { b =>
        ao(b) = new Array(ie)
        cforRange(0 until ie) { i =>
          ao(b)(i) = new Array(je)
          cforRange(0 until je) { j =>
            ao(b)(i)(j) = new Array(ke)
          }
        }
      }

      var c = 0
      cforRange(0 until be) { b =>
        cforRange(0 until ke) { k =>
          cforRange(0 until je) { j =>
            cforRange(0 until ie) { i =>
              ao(b)(i)(j)(k) = ai(c)
              c += 1
            }
          }
        }
      }
      assert(c == ai.length, s"c=$c ai.length ${ai.length}")

      ao
    }

    def toBatchArray2d : Array[Array[Array[Float]]] = {
      val d = x.dim()
      val be = d.batchElems().toInt
      val ao = new Array[Array[Array[Float]]](be)
      val ai = toArray

      val List(ie, je) = dims.take(2).map(_.toInt)
      cforRange(0 until be) { b =>
        ao(b) = new Array(ie)
        cforRange(0 until ie) { i =>
          ao(b)(i) = new Array(je)
        }
      }

      var c = 0
      cforRange(0 until be) { b =>
        cforRange(0 until je) { j =>
          cforRange(0 until ie) { i =>
            ao(b)(i)(j) = ai(c)
            c += 1
          }
        }
      }
      assert(c == ai.length, s"c=$c ai.length ${ai.length}")

      ao
    }

    def toBatchArray1d : Array[Array[Float]] = {
      val d = x.dim()
      val be = d.batchElems().toInt
      assert(d.cols()==1)
      val ao = new Array[Array[Float]](be)
      val ai = toArray

      val List(ie) = dims.take(1).map(_.toInt)
      cforRange(0 until be) { b =>
        ao(b) = new Array(ie)
      }

      var c = 0
      cforRange(0 until be) { b =>
        cforRange(0 until ie) { i =>
          ao(b)(i) = ai(c)
          c+=1
        }
      }
      assert(c == ai.length, s"c=$c ai.length ${ai.length}")

      ao
    }

    def /(y:Expression) : Expression =
      x * Expression.pow(y, -1)

    def str(withVals: Boolean): String =
      if(isBatched){
        if(withVals)   (0 until batchSize).map(i => s"batch elem $i: "+this.pickBatchElem(i).str(withVals)).mkString("\n")
        else           s"Batch($batchSize) of ${this.pickBatchElem(0).str(withVals)}"
      }else if (isScalar) {
        val vals = if(withVals) "["+this.toFloat.toString+"]" else ""
        "Scalar"+vals
      } else {
        val typ  = if (this.isVector) "Vector" else if (this.isMatrix) "Matrix" else "Tensor"
        val dimsStr = this.dims.mkString(", ")
        val vals = if(withVals){
          if(isMatrix){
            val rowSize = dims.head.toInt
            val rows    = this.toSeq.sliding(rowSize, rowSize).toList.transpose
            "\n"+table2str(rows.map(row => row.map(el => f"$el%.2f")))
          }else{
            "["+this.toSeq.mkString(", ")+"]"
          }
        }else{
          ""
        }
        s"$typ($dimsStr)$vals"
      }

    private def table2str(table: Seq[Seq[Any]]) : String = {
      val rows = table.map(row => row.map(_.toString))
      val colWidths = rows.transpose.map(col => col.map(_.length).max+2)
      val rowSeparator = "|"+colWidths.map("-"*_).mkString("|")+"|"
      val header = "+"+colWidths.map("-"*_).mkString("+")+"+"
      val rowStrs = rows.map{row =>
        val s = (row zip colWidths).map{ case (el, width) =>
          " "*(width-el.length-1)+el+" "
        }.mkString("|")
        s"|$s|"
      }
      header+"\n"+rowStrs.mkString("\n"+rowSeparator+"\n")+"\n"+header
    }

    def dims : List[Long] = {
      val d:Dim = x.dim()
      val z = for(i <- 0L until d.size) yield d.get(i)
      val dims = z.toList.reverse.dropWhile(_==1L).reverse  // drops 1s from the back
      if(dims.isEmpty) List(1l) else dims
    }
    def batchSize : Int     = x.dim().batchElems().toInt
    def isBatched : Boolean = this.batchSize > 1
    def isScalar  : Boolean = this.dims == List(1)
    def isVector  : Boolean = this.dims.size == 1 && !isScalar
    def isMatrix  : Boolean = this.dims.size == 2
    def isTensor  : Boolean = this.dims.size >= 3

    def printWithVals(): Unit = System.err.println(this.str(withVals = true ))
    def print()        : Unit = System.err.println(this.str(withVals = false))

    def apply(i: Int     ) : Expression = Expression.pick(x, i)
    def apply(i: Seq[Int]) : Expression = Expression.pick(x, seqInt2UnsignedVector(i), d=0)

    def pickBatchElem(i: Int)          : Expression = Expression.pickBatchElem(x, i)
    def pickRange(from:Int, until:Int) : Expression = Expression.pickrange(x, from, until) // "until" is excluded

    def col(i:Int) : Expression = selectCol(x, i)
    def row(i:Int) : Expression = selectRow(x, i)

  }

}
