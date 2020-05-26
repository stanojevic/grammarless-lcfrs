package edin.supertagger

import edin.nn.DyFunctions._
import edin.nn.model.{IndexedInstance, ModelContainer, String2Int, TrainingController}
import edu.cmu.dynet.{Expression, ParameterCollection}
import edin.algorithms.AutomaticResourceClosing.linesFromFile
import edin.nn.DynetSetup
import edin.nn.contextualized.{SeqOfTagAndWordEmbedder, SeqOfTagAndWordEmbedderConfig}
import edin.nn.layers.{MLP, MLPConfig}
import edin.nn.masking.WordMask

import scala.collection.mutable.ArrayBuffer

class SuperTaggingModel(inTagExtension:String="postags", outTagExtension:String="supertags") extends ModelContainer[TrainInst]{

  import SuperTaggingModel._

  private lazy val useInTags   : Boolean                = hyperParams("main-vars")("in-tag-rep-dim").int > 0
  private lazy val tagMinCount : Int                    = hyperParams("main-vars")("out-tag-min-count").int
  private lazy val tagMaxVoc   : Int                    = hyperParams("main-vars")("out-tag-max-voc"  ).int

  private var allS2I                   : AllS2I                  = _
  private var seqEmbedder              : SeqOfTagAndWordEmbedder = _
  private var inputTagsPredictionModel : SuperTaggingModel       = _
  private var topNN                    : MLP                     = _

  override def toSentences(instance: TrainInst): List[List[String]] = instance.in_words::Nil

  /**
    * called order 2
    * DOC loads training instances
    */
  override def loadTrainData(prefix: String): Iterator[TrainInst] = {
    val allInWords = loadTokens(s"$prefix.words")
    val allOutTags = loadTokens(s"$prefix.$outTagExtension")
    val allInTags  = if(useInTags) loadTokens(s"$prefix.$inTagExtension") else Stream.fill(Integer.MAX_VALUE)(null)
    val res = for{
      ((instInWords, instOutTags), instInTags) <- (allInWords zip allOutTags) zip allInTags
    } yield TrainInst(in_words=instInWords, out_tags=instOutTags, in_tags=instInTags)
    res.iterator
  }

  // DOC removes too long sentences and stuff like that
  override
  def filterTrainingData(trainData:Iterable[IndexedInstance[TrainInst]]) : Iterator[IndexedInstance[TrainInst]] =
    trainData.iterator.filter(_.instance.in_words.nonEmpty)

  /**
    * called order 12
    * DOC saves w2i and similar stuff
    */
  override protected def saveExtra(modelDir: String): Unit =
    allS2I.save(modelDir)

  /**
    * called order 13 // not called during training
    * DOC loads w2i and similar stuff
    */
  override protected def loadExtra(modelDir: String): Unit = {
    if(useInTags){
      inputTagsPredictionModel = new SuperTaggingModel()
      inputTagsPredictionModel.loadFromModelDir(modelDir+"/tagging-model")
    }
    allS2I = AllS2I.load(modelDir)
  }

  /**
    * called order 5
    * DOC defining LSTMs and other parameters
    */
  override protected def defineModelFromHyperFile()(implicit model: ParameterCollection): Unit = {
    val config = hyperParams.mapAllTerminals(Map(
      "RESOURCE_IN_W2I"        -> allS2I.in_w2i,
      "RESOURCE_IN_T2I"        -> allS2I.in_t2i,
      "RESOURCE_OUT_TAGS_SIZE" -> allS2I.out_t2i.size.asInstanceOf[AnyRef],
    ))

    seqEmbedder = SeqOfTagAndWordEmbedderConfig.fromYaml(config("seq-embedder")).construct()
    topNN = MLPConfig.fromYaml(config("MLP")).construct()
  }

  /**
    * called order 4
    * DOC computes s2i and other stuff
    */
  override def prepareForTraining(trainData: Iterable[IndexedInstance[TrainInst]]): Unit = {

    val in_w2i  = new String2Int(minCount=2, maxVacabulary=Int.MaxValue)
    val in_t2i  = new String2Int(minCount=2)
    val out_t2i = new String2Int(minCount=tagMinCount, maxVacabulary=tagMaxVoc, withEOS=false)

    for(IndexedInstance(_, instance) <- trainData){
      for(word <- instance.in_words)
        in_w2i.addToCounts(word)
      for(tag  <- instance.out_tags )
        out_t2i.addToCounts(tag)
      if(useInTags)
        for(auxTag <- instance.in_tags)
          in_t2i.addToCounts(auxTag)
    }
    in_w2i.lock()
    out_t2i.lock()

    System.err.println("you have "+(out_t2i.size-1)+" tags")

    allS2I = new AllS2I(
      in_w2i  = in_w2i,
      in_t2i  = in_t2i,
      out_t2i = out_t2i,
    )
  }

///////////////////////////////////////////////////////////////////////////////////////////////////////

  /**
    * called order 11
    * DOC validates
    */
  override def validate(devData: Iterable[IndexedInstance[TrainInst]]): (Double, Map[String, Double]) = {
    val allInWords   = devData.map(_.instance.in_words).toList
    val allInTags    = devData.map(_.instance.in_tags ).toList
    val allOutTags   = devData.map(_.instance.out_tags).toList
    var totalCorrect = 0d
    var totalAll     = 0d

    for(((predTagsRaw, words), goldTags) <- decodeLotsOfSentsByBatching(allInWords, allInTags, 1) zip allInWords zip allOutTags){
      val predTags = predTagsRaw.map(_.head._1)
      assert(predTags.size == goldTags.size, s"lengths don't match\nwords:\t$words\ngold:\t$goldTags\npred:\t$predTags")
      totalCorrect += (predTags zip goldTags).count{case (x, y) => x==y}
      totalAll     += predTags.size
    }
    val precision = totalCorrect/totalAll
    (precision, Map("precision"->precision))
  }

  override def computeLossForBatch(batch: Seq[IndexedInstance[TrainInst]]): Expression = {
    val batchInWords      = batch.map(_.instance.in_words).toList
    val batchInTags       = if(useInTags) batch.map(_.instance.in_tags).toList else null
    val maxLen            = batchInWords.map(_.size).max
    val batchOutTags      = for(IndexedInstance(_, TrainInst(_, tags, _)) <- batch) yield tags.map(allS2I.out_t2i(_))++List.fill(maxLen-tags.size)(0)
    val (scores, masks)   = findLogSoftmaxes(batchInWords, batchInTags)
    val scores2           = for((score, mask        ) <-  scores  zip masks               ) yield mask.fillMask(score, 0f)
    val scores3           = for((score, tagsVertical) <- (scores2 zip batchOutTags.transpose)) yield score(tagsVertical)
//    System.err.print("."*batch.size)
    -scores3.map(_.sumBatches).esum/batch.size
  }

  def tagManySentArgmax(inWords:List[List[String]]) : List[List[String]] = {
    val inTags = if(useInTags) inputTagsPredictionModel.tagManySentArgmax(inWords) else List.fill(inWords.size)(null)
    decodeLotsOfSentsByBatching(inWords, inTags, 1).map(_.map(_.head._1))
  }

  def tagSingleSentArgmax(sent:List[String]) : List[String] =
    tagManySentArgmax(List(sent)).head

  def decodeLotsOfSentsBySliding(allInWords: List[List[String]], allInTags: List[List[String]], k:Int) : List[List[List[(String, Float)]]] = {
    val res = ArrayBuffer[List[List[(String, Float)]]]()
    val batches = (allInWords zip allInTags).sliding(1, 1)
    for((batch, batchId) <- batches.zipWithIndex){
      System.err.println(s"processing $batchId")
      DynetSetup.cg_renew()
      val inWordsBatch = batch.map(_._1)
      val inTagsBatch  = batch.map(_._2)
      for((sent, sentRes) <- batch zip decodeBatch(inWordsBatch, inTagsBatch, k))
        res.append(sentRes)
    }
    res.toList
  }

  def decodeLotsOfSentsByBatching(allInWords: List[List[String]], allInTags: List[List[String]], k:Int) : List[List[List[(String, Float)]]] = {
    val res = scala.collection.mutable.Map[List[String], List[List[(String, Float)]]]()
    val batches = TrainingController.makeBatchesByWordCount((allInWords zip allInTags), allInWords.map(_.size), 3000)
    for((batch, batchId) <- batches.zipWithIndex){
//      System.err.println(s"processing $batchId")
      DynetSetup.cg_renew()
      val inWordsBatch = batch.map(_._1)
      val inTagsBatch  = batch.map(_._2)
      for((sent, sentRes) <- inWordsBatch zip decodeBatch(inWordsBatch, inTagsBatch, k))
        res(sent) = sentRes
    }
    allInWords.map(res)
  }

  private def decodeBatch(batchInWords: List[List[String]], batchInTags: List[List[String]], k:Int) : List[List[List[(String, Float)]]] = {
    val (exps, masks: Seq[WordMask]) = findLogSoftmaxes(batchInWords, batchInTags)
    val allScores = (exps zip masks).map{case (e, m) => m extractWordValuesVertical e}.transpose.map(_.flatten)
    for(sentScore <- allScores) yield
      for(wordScore <- sentScore) yield {
        val bestScores = argmaxWithScores(wordScore, k+1).filter(_._1!=allS2I.out_t2i.UNK_i).take(k) // +1 is needed in case UNK is predicted
        for((tagId, tagScore) <- bestScores) yield
          (allS2I.out_t2i(tagId), tagScore)
      }
  }

  private def findLogSoftmaxes(batchInWords: List[List[String]], batchInTags: List[List[String]]) : (List[Expression], List[WordMask]) = {
    val (exps, masks) = seqEmbedder.transduceBatch(batchInWords, batchInTags)
    (exps map topNN.apply, masks)
  }

}

object SuperTaggingModel {

  def loadTokens(fn: String): Iterable[List[String]] =
    linesFromFile(fn).map(_.split(" +").toList).toList

}

