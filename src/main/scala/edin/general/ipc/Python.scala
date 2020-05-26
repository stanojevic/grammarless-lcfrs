package edin.general.ipc

import java.io.File
import java.util

import edin.general.Global
import jep.{Jep, JepConfig, NDArray}

import scala.collection.JavaConverters._

object Python {

  private val TMP = "tmp_var_name"
  private var varNameCount = 0
  def newVarName : String = {
    varNameCount += 1
    TMP+varNameCount
  }

  def addToPath(dir:String) : Unit = {
    importt("sys")
    set(TMP, dir)
    exec(s"sys.path.insert(0, $TMP)")
  }

  def runPythonFile(fn:String) : Unit = {
    addToPath(new File(fn).getParent)
    jep.runScript(fn)
  }

  def activeVars() : List[String] =
    getListOfStrings("dir()").
      filterNot(x => x.startsWith("__") && x.endsWith("__")).
      filterNot(x => x=="jep").
      filterNot(x => imported contains x)

  def delActiveVars() : Unit = activeVars().foreach(delVar)

  def delVar(varr:String) : Unit = exec(s"del $varr")

  def exec(cmd:String*) : Unit = jep.eval(cmd.mkString("\n"))

  def set(varr:String, vall:Any) : Unit = jep.set(varr, vall)

  def setList(varr:String, vall:Seq[_], freeTMP:String=TMP) : Unit = {
    exec(s"$varr = []" )
    for(x <- vall){
      x match {
        case a:Seq[_] => setList(freeTMP, a, freeTMP+"x")
        case _        => set(freeTMP, x)
      }
      exec(s"$varr.append($freeTMP)")
      delVar(freeTMP)
    }
  }

  def getListOfStrings(varr:String) : List[String] = jep.getValue(varr).asInstanceOf[util.ArrayList[String]].asScala.toList

  def getListOfLongs(varr:String) : List[Long] = jep.getValue(varr).asInstanceOf[util.ArrayList[Long]].asScala.toList

  def getListOfDoubles(varr:String) : List[Double] = jep.getValue(varr).asInstanceOf[util.ArrayList[Double]].asScala.toList

  def getListOfNumPyArrays(varr:String) : List[Array[Float]] = jep.getValue(varr).asInstanceOf[util.ArrayList[NDArray[Array[Float]]]].asScala.toList.map(_.getData)

//  def getLong(varr:String) : Long = {
//    exec(s"$TMP = $varr")
//    System.err.println(jep.getValue(TMP))
//    jep.getValue(TMP).asInstanceOf[Long]
////    jep.getValue(varr).asInstanceOf[Long]
//  }
  def getLong(varr:String) : Long = jep.getValue(varr).asInstanceOf[Long]

  def getInt(varr:String) : Int = getLong(varr).toInt

  def getDouble(varr:String) : Double = jep.getValue(varr).asInstanceOf[Double]

  def getString(varr:String) : String = jep.getValue(varr).asInstanceOf[String]

  def getNumPyDouble(varr:String) : Double = jep.getValue(s"np.asscalar($varr)").asInstanceOf[Double]

  def setNumPyArray(varr:String, vall:Array[Float]) : Unit = jep.set(varr, new NDArray[Array[Float]](vall, vall.length))

  def getNumPyArray(cmd:String) : Array[Float] = jep.getValue(cmd).asInstanceOf[NDArray[Array[Float]]].getData

  private var imported = Set[String]()
  def importt(module:String) : Unit = {
    imported += module
    exec(s"import $module")
  }
  def importtAs(module:String, as:String) : Unit = {
    imported += as
    exec(s"import $module as $as")
  }
  def importt(from:String, module:String) : Unit = {
    imported += module
    exec(s"from $from import $module")
  }
  def importtAs(from:String, module:String, as:String) : Unit = {
    imported += as
    exec(s"from $from import $module as $as")
  }
  def importNumPy() : Unit = importtAs("numpy", "np")

  def apply(cmds:String*) : Unit = exec(cmds:_*)

  def jep: Jep = {
    if(jepInstance == null){
      val JEP_DIR = Global.projectDir+"/lib"
      if(! new File(JEP_DIR).exists())
        throw new Exception(s"jep dir $JEP_DIR not found!")
      val jepConfig = new JepConfig()
        .setInteractive(false)
        .setIncludePath(JEP_DIR)
        .setClassLoader(null)
        .setClassEnquirer(null)
      jepInstance = new Jep(jepConfig)
      importt("sys")
      exec("sys.argv = ['program_name'] ")
    }
    jepInstance
  }

  def closeJep() : Unit = {
    if(jepInstance != null)
      jepInstance.close()
  }

  override protected def finalize(): Unit = { closeJep() }

  private var jepInstance:Jep = _

}
