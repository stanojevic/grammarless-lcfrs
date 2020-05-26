package edin.nn.contextualized

import java.io._

import edin.general.StringMapDatabase
import edin.general.ipc.Python
import edin.nn.model.TrainingController
import scala.collection.mutable.{Map => MutMap}

object SequenceEmbedderPrecomputable{

  private var seqEmbPrecomputers : List[SequenceEmbedderPrecomputable] = List()

  private def registerSeqEmb(semb:SequenceEmbedderPrecomputable) : Unit =
    if(! seqEmbPrecomputers.exists(_.name == semb.name))
      seqEmbPrecomputers ::= semb

  // does precomputation for all registered precomputers
  def precompute(sents: => List[List[String]], fn:String) : Unit =
    seqEmbPrecomputers.foreach(_.precompute(sents, fn))

  // does precomputation only for a particular precompter called embName
  def precomputeParticular(embName:String, sents: => List[List[String]], fn:String) : Unit =
    seqEmbPrecomputers.find(_.name==embName).foreach(_.precompute(sents, fn))

  def closePrecomputingPart() : Unit =
    seqEmbPrecomputers.foreach(_.closePrecomputing())

}

trait SequenceEmbedderPrecomputable{

  private type Sent = List[String]

  // TO OVERRIDE type of the embeddings that will be stored, for example List[Array[Float]] for layers of vectors
  protected       type WordEmbType <: AnyRef
  protected final type SentEmb = List[WordEmbType]

  // used during trianing only to close the preembedding part when it's not needed any more (for instance, python)
  protected def closePrecomputing() : Unit

  // OPTIONAL OVERRIDE how big should be the chunks sent for precomputation
  protected val precomputingBatchSizeInWord = 3000
  // OPTIONAL OVERRIDE how big should be the sequences before they are broken into two
  protected val precomputingMaxSeqSplit     =   90

  // TO OVERRIDE actual mini-batch embedder
  protected def embedBatchDirect(sents: Seq[Sent]) : List[SentEmb]

  // TO OVERRIDE name that will be used to find the embedder
  val name: String

//  private   final var cache:MutMap[String, Emb] = _
  private final val caches = MutMap[String, StringMapDatabase[SentEmb]]()
  private final def lookupCache(sent: List[String]) : Option[SentEmb] = caches.values.toStream.flatMap(_.get(sent mkString " ")).headOption

  SequenceEmbedderPrecomputable.registerSeqEmb(this)

  protected final def findSentsEmb(sents:List[Sent]) : List[SentEmb] = {
    val (preComputed, toCompute) = sents.map(sent => (sent, lookupCache(sent))).zipWithIndex.partition(_._1._2.nonEmpty)
    val res1: Seq[(SentEmb, Int)] = preComputed.map{case ((sent, Some(emb)), i) => (emb, i)}
    val res2: Seq[(SentEmb, Int)] = if(toCompute.isEmpty) Nil else handleBatchPrecomputation(toCompute.map(_._1._1)) zip toCompute.map(_._2)
    (res1 ++ res2).sortBy(_._2).map(_._1).toList
  }

  private final def precompute(sents: => List[Sent], fn: String): Unit = {

    Python.exec("")

    val file = new File(fn+"."+name)
    val lockFile = new File(fn+"."+name+".lock")

    if(lockFile.exists()){
      System.err.println(s"I believe some other process is precomputing embeddings so I'm waiting for it to finish")
      System.err.println(s"if that is not the case:")
      System.err.println(s" - kill this process")
      System.err.println(s" - delete $file")
      System.err.println(s" - delete $lockFile")
      System.err.println(s" - start this process again")
      var i = 0
      while(lockFile.exists()){
        if(i%30==0)
          System.err.println(s"waiting for $name embeddings ; 5 minute check")
        Thread.sleep(10*1000)
        i+=1
      }
    }
    if(! file.exists()){
      System.err.println(s"Precomputing $name START")
      lockFile.createNewFile()
      val cacheFileMap = StringMapDatabase.create[SentEmb](file.getAbsolutePath, readOnly=false)

      var sentCount = 0
      val sentTotal = sents.size
      for(batch <- TrainingController.makeBatchesByWordCount(sents, sents.map(_.size), precomputingBatchSizeInWord)){
        sentCount += batch.size
        System.err.println(f"Precomputed $sentCount%5d/$sentTotal\t\tcurrent batch with ${batch.size} sents with lengths in range ${(batch.map(_.size).min, batch.map(_.size).max)}")
        for((sent, embs) <- batch zip handleBatchPrecomputation(batch)){
          cacheFileMap += sent.mkString(" ") -> embs
        }
      }

      cacheFileMap.close()
      lockFile.delete()
      System.gc()
      System.err.println(s"Precomputing $name END")
    }
    caches += file.getAbsolutePath -> StringMapDatabase.create(file.getAbsolutePath, readOnly=true)
  }

  // does sentence splitting in case sentences are too long
  private def handleBatchPrecomputation(batch:List[Sent]) : List[SentEmb] = {
    val splitted = batch.flatMap{ sent =>
      val parts = sent.sliding(precomputingMaxSeqSplit, precomputingMaxSeqSplit).toList
      parts.init.map((0, _)) :+ (1, parts.last)
    }
    val embedded = embedBatchDirect(splitted.map(_._2))
    (splitted zip embedded).foldLeft((List[WordEmbType](), List[SentEmb]())){
      case ((curr, all), ((0, _), emb)) =>
        (curr++emb,              all)
      case ((curr, all), ((1, _), emb)) =>
        (      Nil, curr++emb :: all)
    }._2.reverse
  }

}

