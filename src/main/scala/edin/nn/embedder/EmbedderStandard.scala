package edin.nn.embedder

import edin.nn.DyFunctions._
import edin.algorithms.MathArray
import edin.nn.{DyFunctions, DynetSetup}
import edu.cmu.dynet.{Expression, FloatVector, LookupParameter, ParameterCollection}
import edin.algorithms.AutomaticResourceClosing.linesFromFile
import edin.nn.layers.{Dropout, IdentityLayer, Layer, SingleLayer}
import edin.nn.model.{Any2Int, String2Int, TrainingController, YamlConfig}

import scala.util.Random

sealed case class EmbedderStandardConfig[K](
                                             s2i                : Any2Int[K],
                                             outDim             : Int,
                                             dropout            : Float=0f,
                                             initWithPretrained : String=null,
                                             wordDropoutUse     : Boolean = false,
                                             wordDropoutAlpha   : Double = 0.25
                                   ) extends EmbedderConfig[K] {
  def construct()(implicit model: ParameterCollection) = new EmbedderStandard[K](this)
}

object EmbedderStandardConfig{

  def fromYaml[K](conf:YamlConfig) : EmbedderConfig[K] =
    EmbedderStandardConfig[K](
      s2i                 = conf("w2i").any2int,
      outDim              = conf("out-dim").int,
      initWithPretrained  = conf.getOrElse[String]("init-with", null),
      dropout             = conf.getOrElse("dropout", 0f),
      wordDropoutUse      = conf.getOrElse("word-dropout-use", false),
      wordDropoutAlpha    = conf.getOrElse("word-dropout-alpha", 0.25)
    )

}

class EmbedderStandard[T](config:EmbedderStandardConfig[T])(implicit model: ParameterCollection) extends Embedder[T] {

  var s2i:Any2Int[T] = config.s2i
  private val dropProb:Float = config.dropout

  override val outDim = config.outDim
  private  val drop   = Dropout(config.dropout)

  private val (eTable: LookupParameter, compressor:Layer) = if(config.initWithPretrained != null && TrainingController.modelDefinitionDuringTraining){
    val varName = config.initWithPretrained.toLowerCase match {
      case "glove" => "GLOVE_FILE"
      case "fasttext" => "FASTTEXT_FILE"
    }
    if(! sys.env.contains(varName))
      sys.error(s"you have to put location of embeddings file in $$$varName environment variable")
    val fileName = sys.env(varName)
    val subDim = EmbedderStandard.pretrainedEmb_loadDim(fileName)
    val E = model.addLookupParameters(s2i.size, subDim)
    EmbedderStandard.initEmbedderFromPretrainedTable(fileName, E, s2i.asInstanceOf[Any2Int[String]], false)
    (E, SingleLayer.compressor(subDim, outDim))
  }else{
    (model.addLookupParameters(s2i.size, outDim), IdentityLayer())
  }

  private var droppedWordTypes = Set[T]()
  private var latestCG  : Int = -1

  private def shouldWordDrop(w:T, alpha:Double, s2i:Any2Int[T]) : Boolean = {
    /** used in the similar way to Kiperwasser & Goldberg 2016 but over word types as in Gal & Ghahramani 2016 */
    if(latestCG != DynetSetup.cg_id){
      latestCG = DynetSetup.cg_id
      droppedWordTypes = Set()
    }

    if(droppedWordTypes contains w){
      true
    }else{
      val counts = s2i.frequency(w)
      val dropProb = config.wordDropoutAlpha / (counts + config.wordDropoutAlpha)
      val willDrop = Random.nextDouble() < dropProb
      if(willDrop){
        droppedWordTypes += w
      }
      willDrop
    }
  }

  override def apply(xs: List[T]): Expression = {
    val xs2 = xs.map{ w =>
      if(config.wordDropoutUse && DyFunctions.dropoutIsEnabled && shouldWordDrop(w, config.wordDropoutAlpha, s2i)){
        s2i.UNK_str
      }else{
        w
      }
    }
    drop(compressor(eTable(xs2 map s2i.apply)))
  }

}

object EmbedderStandard{

  private def linesFromEmbFile(fn:String) : Iterator[String] =
    linesFromFile(fn, "UTF-8").dropWhile(_.split(" +").length==2) // this part is needed for FastText format

  def pretrainedEmb_loadDim(file:String) : Int =
    linesFromEmbFile(file).next().split(" +").length-1

  def pretrainedEmb_loadS2I(file:String, lowercased:Boolean) : Any2Int[String] = {
    val s2i : Any2Int[String] = new String2Int(lowercased = lowercased)
    linesFromEmbFile(file).zipWithIndex.foreach{ case (line, i) =>
      val word = line.split(" ")(0)
      s2i.addToCounts(word)
      if(i% 100000 == 0)
        System.err.println(s"Loading w2i for pretrained embeddings $i")
    }
    s2i
  }

  def initEmbedderFromPretrainedTable(fn:String, E:LookupParameter, s2i:Any2Int[String], normalized:Boolean) : Unit = {
    var avgVector:MathArray = null
    var vecCount = 0
    linesFromEmbFile(fn).zipWithIndex.foreach{ case (line, line_id) =>
      vecCount+=1
      val fields = line.split(" ").toList
      val word = fields.head
      val i = s2i(word)
      if(i!=s2i.UNK_i){
        var v = new MathArray(fixEmbedding(fields.tail.map{_.toFloat}.toArray))
        if(normalized)
          v = new MathArray(to_l2_normalized(v.array))
        if(avgVector == null){
          avgVector = MathArray(v.length)
        }
        avgVector += v
        val vec:FloatVector = new FloatVector(v.toArray)
        E.initialize(i, vec)
      }
      if(line_id % 100000 == 0)
        System.err.println(s"Loading pretrained embedding vectors file line $line_id")
    }
    avgVector /= vecCount
    E.initialize(s2i.UNK_i, new FloatVector(avgVector.toArray))
    E.setUpdated(false)
  }

  private def to_l2_normalized(array:Array[Float]) : Array[Float] = {
    var norm = array.map(x => x*x).sum
    if(norm == 0) norm = 1
    array.map(_/norm)
  }

  private def fixEmbedding(emb:Array[Float]) : Array[Float] = {
    val e = emb.clone()
    for(i <- e.indices){
      if(e(i).isNaN || e(i).isInfinity){
        System.err.println("EMBEDDING IS WRONG; I'M FIXING IT")
        e(i) = 0f
      }
    }
    e
  }

}


