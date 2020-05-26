package edin.mcfg

import java.io.File

import edu.cmu.dynet.{Dim, Expression, ParameterCollection}
import spire.syntax.cfor.cforRange
import edin.algorithms.evaluation.FScore
import edin.constituency.representation.ConstNode
import edin.nn.DyFunctions._
import edin.nn.DynetSetup
import edin.nn.attention.{WordPairScorer, WordPairScorerConfig}
import edin.nn.contextualized.{SeqOfTagAndWordEmbedder, SeqOfTagAndWordEmbedderConfig}
import edin.nn.layers.{Layer, MLPConfig}
import edin.nn.model.{IndexedInstance, ModelContainer, String2Int}
import edin.supertagger.SuperTaggingModel

import scala.collection.immutable.BitSet

class ParsingModel(
                    var maxDiscElSize                         : Int     = -1,
                    var maxGapSize                            : Int     = -1,
                    var maxSentenceSizeForDicontinuousParsing : Int     = 40,
                    var spanBreakageScoring                   : Boolean = false
                  ) extends ModelContainer[ConstNode] {

  private var rulesToUse             : BitSet                   = BitSet()

  private var allS2I                 : AllS2I                   = _

  private var seqEmbedder            : SeqOfTagAndWordEmbedder  = _
  private var leftNtFF               : Layer                    = _
  private var rightNtFF              : Layer                    = _
  private var biaffine               : WordPairScorer           = _

  private var taggingModel           : SuperTaggingModel        = _

  private var isGlobalModel          : Boolean                  = true
  private var isExplicitBinarization : Boolean                  = true

  /**
    * called order 2
    * DOC loads training instances
    */
  override def loadTrainData(prefix: String): Iterator[ConstNode] =
    ConstNode.loadTreesFromFile(prefix).iterator

  /**
    * called order 4
    * DOC computes s2i and other stuff
    */
  override def prepareForTraining(trainData: Iterable[IndexedInstance[ConstNode]]): Unit = {
    val w2i = new String2Int(minCount = 2)
    val t2i = new String2Int()
    val n2i = new String2Int(withUNK = false)
    val coreNT = scala.collection.mutable.Set[String]()
    for(IndexedInstance(_, tree) <- trainData){
      for(leaf <- tree.leafs){
        w2i.addToCounts(leaf.word)
        t2i.addToCounts(leaf.label)
      }
      for(node <- foldTree(tree).allNodes){
        n2i.addToCounts(node.label)
        n2i.addToCounts(node.label+"__1")
        n2i.addToCounts(node.label+"__2")
        n2i.addToCounts(node.label+"__3")
        coreNT += node.label
      }
    }
    coreNT += "∅"
    n2i.addToCounts("∅")
    n2i.addToCounts("∅__1")
    n2i.addToCounts("∅__2")
    n2i.addToCounts("∅__3")
    val coreNTdesc = coreNT.toList.map(nt => Array(n2i(nt), n2i(nt+"__1"), n2i(nt+"__2"), n2i(nt+"__3"))).toArray
    allS2I = new AllS2I(
      w2i        = w2i,
      t2i        = t2i,
      n2i        = n2i,
      coreNTdesc = coreNTdesc,
    )
  }

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
    allS2I = AllS2I.load(modelDir)
    val taggingDir = modelDir+"/tagging-model"
    if(new File(taggingDir).exists()){
      taggingModel = new SuperTaggingModel()
      taggingModel.loadFromModelDir(taggingDir)
    }else{
      System.err.println(s"couldn't load $taggingDir")
    }
  }

  /**
    * called order 5
    * DOC defining LSTMs and other parameters
    */
  override protected def defineModelFromHyperFile()(implicit model: ParameterCollection): Unit = {
    val hyperParams = this.hyperParams.mapAllTerminals(Map(
      "RESOURCE_W2I"             -> allS2I.w2i,
      "RESOURCE_T2I"             -> allS2I.t2i,
      "RESOURCE_NONTERMS_NUMBER" -> allS2I.n2i.size.asInstanceOf[AnyRef],
    ))

    isExplicitBinarization = hyperParams("main-vars")("explicit-binarization").bool
    isGlobalModel          = hyperParams("main-vars")("global-model").bool
    require(isExplicitBinarization || isGlobalModel, "you can't have implicit binarization with probabilistic model")

    leftNtFF        = MLPConfig.fromYaml(hyperParams("ff-left" )).construct()
    rightNtFF       = MLPConfig.fromYaml(hyperParams("ff-right")).construct()

    seqEmbedder     = SeqOfTagAndWordEmbedderConfig.fromYaml(hyperParams("seq-embedder")).construct()

    biaffine        = WordPairScorerConfig.fromYaml(hyperParams("pair-seq-labels")).construct()
    if(this.rulesToUse.isEmpty){
      setRulesToUse( hyperParams("main-vars")("rules-to-use-for-valid").str )
    }
    if(maxDiscElSize== -1){
      maxDiscElSize = hyperParams("main-vars").getOrElse("max-disc-el-size", 1000)
    }
    if(maxGapSize== -1){
      maxGapSize = hyperParams("main-vars").getOrElse("max-gap-size", 1000)
    }
  }

  def setRulesToUse(rulesType:String) : Unit = {
    this.rulesToUse = rulesType match {
      case "MCFG"          => ParsingAlgorithm.rulesMCFG
      case "CFG"           => ParsingAlgorithm.rulesCFG
      case "wnMCFG"        => ParsingAlgorithm.rulesWN
      case "wnMCFGnonRare" => ParsingAlgorithm.rulesWNnonRare
    }
  }

  private def combineSpans(spans1:List[(Int, Int)], spans2:List[(Int, Int)]) : List[(Int, Int)] =
    (spans1 ++ spans2).sortBy(_._1).map(List(_)).reduce{ (xs, ys) =>
      val x = xs.last
      val y = ys.head

      if(x._2 == y._1)
        (xs.init++((x._1, y._2)::ys.tail))
      else
        xs ++ ys
    }

  /**
    * called order 10
    * DOC computes loss per instance after everything is ready
    */
  override def computeLossForInstance(instance:IndexedInstance[ConstNode]) : Expression =
    if(isGlobalModel)
      globalLoss(instance.instance)
    else
      probLoss(instance.instance)

  private def getProbScores(words:List[String], tags:List[String]) : (Expression, Expression) = {
    val scores = getRawScores(words, tags)
    val n = words.size
    val l = biaffine.labels

    val scoresGroup  = logSoftmax(
      concatByDim(
        List(scores, zeros(scores.dim())),
        d=3
      ).  // (x, y, l, 2), b
      reshape(n*n*l, 2). // (x*y*l, 2), b
      T. // (2, x*y*l), b
      reshape(Dim(List(2), b=n*n*l)) // (2), x*y*l*b
    )
    val scoresYes = scoresGroup(0).reshape((n, n, l)) // (x, y, l)
    val scoresNo  = scoresGroup(1).reshape((n, n, l)) // (x, y, l)

    (scoresYes, scoresNo)
  }

  private def foldTree(tree:ConstNode) : ConstNode = {
    val tree1 = if(isExplicitBinarization) tree.binarizedLossy else tree
    val tree2 = Transforms.transformFold(tree1)
    tree2
  }

  private def unfoldTree(words:List[String], tags:List[String], tree:ConstNode) : ConstNode =
    Transforms.transformUnfold(words, tags, tree)

  private def probLoss(goldTreeOrig:ConstNode) : Expression = {
    val goldTreeFolded = foldTree(goldTreeOrig)
    val words = goldTreeFolded.leafsSorted.map(_.word )
    val tags  = goldTreeFolded.leafsSorted.map(_.label)
    val (yesLogProbs, noLogProbs) = getProbScores(words, tags)

    val (maskConstIsOne, maskGoodIsOne) = findMasks(goldTreeFolded)
    val maskBadIsOne = maskConstIsOne⊙(1-maskGoodIsOne)

    val logProb = sumElems(maskGoodIsOne⊙yesLogProbs + maskBadIsOne⊙noLogProbs)
    -logProb
  }

  private def findMasks(treeFolded: ConstNode) : (Expression, Expression) = {
    val n    = treeFolded.words.size
    val ls   = allS2I.n2i.size

    def constructMask() : Array[Array[Array[Float]]] = {
      val a = new Array[Array[Array[Float]]](n)
      cforRange(0 until n) { i =>
        a(i) = new Array(n)
        cforRange(0 until n) { j =>
          a(i)(j) = new Array(ls)
        }
      }
      a
    }

    val maskConstIsOne = constructMask()
    cforRange(0 until n)  { i =>
      cforRange(i until n) { j =>
        maskConstIsOne(i)(j) = Array.fill(ls)(1f)
      }
    }

    val maskGoodIsOne = constructMask()

    val n2i = allS2I.n2i
    for(node <- treeFolded.allNodes){
      node.spans match {
        case List((a, b)) =>
          maskGoodIsOne(a)(b-1)(n2i(node.label)) = 1f
        case List((a, b), (c, d)) =>
          maskGoodIsOne(a)(b-1)(n2i(node.label+"__1")) = 1f
          maskGoodIsOne(b)(c-1)(n2i(node.label+"__2")) = 1f
          maskGoodIsOne(c)(d-1)(n2i(node.label+"__3")) = 1f
        case _ =>
      }
      if(node.children.size>2){
        for{
          c1 <- node.children
          c2 <- node.children
          if c1.spans != c2.spans
          spans = combineSpans(c1.spans, c2.spans)
        } spans match {
            case List((a, b)) =>
              maskGoodIsOne(a)(b-1)(n2i("∅")) = 1f
            case List((a, b), (c, d)) =>
              maskGoodIsOne(a)(b-1)(n2i("∅__1")) = 1f
              maskGoodIsOne(b)(c-1)(n2i("∅__2")) = 1f
              maskGoodIsOne(c)(d-1)(n2i("∅__3")) = 1f
            case _ =>
        }
      }
    }

    (tensor(maskConstIsOne), tensor(maskGoodIsOne))
  }

  private def globalLoss(goldOrig: ConstNode): Expression = {
    val goldFold = foldTree(goldOrig)
    val words    = goldOrig.leafsSorted.map(_.word )
    val tags     = goldOrig.leafsSorted.map(_.label)
    val n        = words.size

    val scores   = getRawScores(words, tags)

    val correctLabels = goldFold.allNodes.flatMap{ node =>
      var corrects = node.spans match {
        case List((a, b))         => (allS2I.n2i(node.label), (a, b, -1, -1)) :: Nil
        case List((a, b), (c, d)) => (allS2I.n2i(node.label), (a, b,  c,  d)) :: Nil
        case _                    => Nil
      }
      if(node.children.size>2){
        val childrenSpans = node.children.map(_.spans)
        for{
          child1Span <- childrenSpans
          child2Span <- childrenSpans
          if child1Span!=child2Span
          combinedSpans = combineSpans(child1Span, child2Span)
          if combinedSpans.size <= 2
        }{
          combinedSpans match {
            case List((a, b))         => corrects ::= (-1, (a, b, -1, -1))
            case List((a, b), (c, d)) => corrects ::= (-1, (a, b,  c,  d))
          }
        }
      }
      corrects
    }

    val (predFold, predParseScore, _)  = try{parseFolded(scores.toArray3d, correctLabels)}catch{case e:Exception => System.err.println(s"failed for sentence with:\nwords: ${words mkString " "}\ntags: ${tags mkString " "}") ; throw e}
    val goldScore = treeScore(goldFold, scores)
    val predScore = treeScore(predFold, scores)
    val margin    = eval(goldOrig, unfoldTree(words, tags, predFold))._2("hamming_err_avg")

    if(predScore.toFloat + margin > goldScore.toFloat){
      predScore + margin - goldScore
    }else{
      scalar(0f)
    }
  }

  private def treeScore(treeFolded:ConstNode, scores:Expression) : Expression =
    treeFolded.allNodes.filterNot(_.spans.size>2).map{ node =>
      val coreLabelStr = node.label
      val coreLabel    = allS2I.n2i(coreLabelStr)
      val l1           = allS2I.n2i(coreLabelStr+"__1")
      val l2           = allS2I.n2i(coreLabelStr+"__2")
      val l3           = allS2I.n2i(coreLabelStr+"__3")
      node.spans match {
        case List((a, b))         =>                      scores(a)(b-1)(coreLabel)
        case List((a, b), (c, d)) => scores(a)(b-1)(l1) + scores(b)(c-1)(l2)       + scores(c)(d-1)(l3)
      }
    }.esumOrElse(scalar(0))

  private def getRawScores(words:List[String], tags:List[String]) : Expression = {
    val n   = words.size
    val l   = biaffine.labels
    val ees = seqEmbedder.transduce(words, tags)
    val els = ees map leftNtFF.apply
    val ers = ees map rightNtFF.apply
    biaffine(els, ers).reshape((n*l, n)).T.reshape((n, n, l)) // (x, y, l), b
  }

  private def predictTags(words:List[String]) : List[String] = {
    require(taggingModel != null, "tagging model was not loaded!")
    taggingModel.tagSingleSentArgmax(words)
  }

  def parse(words:List[String]) : (ConstNode, (Double, Double, Double, Double)) =
    parse(words, predictTags(words))

  def parseOracle(goldOrigIn:ConstNode) : ConstNode = {

    def removeFanOut(maxFanOut:Int, node:ConstNode) : ConstNode = {
      val newChildren = for{
        oldChild <- node.children
        newChild =  removeFanOut(maxFanOut, oldChild)
        subChild <- if(newChild.spans.size>maxFanOut) newChild.children else List(newChild)
      } yield subChild
      node.copy(children = newChildren.sortBy(_.indices.head))
    }
    val goldOrig = if(this.rulesToUse == BitSet(3)) removeFanOut(1, goldOrigIn) else removeFanOut(2, goldOrigIn)

    val words = goldOrig.leafsSorted.map(_.word )
    val tags  = goldOrig.leafsSorted.map(_.label)
    val goldFold = foldTree(goldOrig)
    val n  = words.size
    val ls = allS2I.n2i.size

    val table: Array[Array[Array[Float]]] = new Array(n)
    for(i <- 0 until n){
      table(i) = new Array(n)
      for(j <- 0 until n){
        table(i)(j) = Array.fill(ls)(-10000)
      }
    }

    for(node <- goldFold.allNodes){
      val coreLabelStr = node.label
      if(node.spans.size<=2 && allS2I.n2i.contains(coreLabelStr)){
        val coreLabel    = allS2I.n2i(coreLabelStr)
        val l1           = allS2I.n2i(coreLabelStr+"__1")
        val l2           = allS2I.n2i(coreLabelStr+"__2")
        val l3           = allS2I.n2i(coreLabelStr+"__3")
        node.spans match {
          case List((a, b))         =>
            table(a)(b-1)(coreLabel)=0
          case List((a, b), (c, d)) =>
            table(a)(b-1)(l1)=0
            table(b)(c-1)(l2)=0
            table(c)(d-1)(l3)=0
        }
      }
    }

    unfoldTree(words, tags, parseFolded(table, null)._1)
  }

  private def getScoresForParsing(words:List[String], tags:List[String]) : Array[Array[Array[Float]]] =
    if(isGlobalModel){
      getRawScores(words, tags).toArray3d
    }else{
      val (scoresYesE, scoresNoE) = getProbScores(words, tags)
      val scoresYes = scoresYesE.toArray3d

      if(spanBreakageScoring){
        val scoresNo  = scoresNoE.toArray3d

        val n = scoresYes.length
        val spanOff = new Array[Array[Float]](n)
        cforRange(0 until n){i => spanOff(i) = new Array(n)}
        val leftBreak = new Array[Array[Float]](n)
        cforRange(0 until n){i => leftBreak(i) = new Array(n)}
        val rightBreak = new Array[Array[Float]](n)
        cforRange(0 until n){i => rightBreak(i) = new Array(n)}

        for(l <- allS2I.n2i.all_non_UNK_values if ! l.endsWith("_2")){
          val li = allS2I.n2i(l)
          cforRange(0 until n-1){ i =>
            cforRange(i+1 until n) { j =>
              spanOff(i)(j) += scoresNo(i)(j)(li)
            }
          }
        }

        cforRange(0 until n-1){ i =>
          cforRange(i+1 until n) { j =>
            leftBreak(i)(j) = leftBreak(i)(j-1)
            cforRange(0 until i-1){ k =>
              leftBreak(i)(j) += spanOff(k)(j-1)
            }
            rightBreak(i)(j) = rightBreak(i+1)(j)
            cforRange(j+1 until n){ k =>
              rightBreak(i)(j) += spanOff(i+1)(k)
            }
          }
        }

        for(l <- allS2I.n2i.all_non_UNK_values if ! l.endsWith("_2")){
          val li = allS2I.n2i(l)
          cforRange(0 until n-1) { i =>
            cforRange(i + 1 until n) { j =>
              scoresYes(i)(j)(li) += leftBreak(i)(j) + rightBreak(i)(j)
            }
          }
        }
      }

      scoresYes
    }

  def parse(words:List[String], tags:List[String]) : (ConstNode, (Double, Double, Double, Double)) = {
    val time1 = System.nanoTime()
    val scores = getScoresForParsing(words, tags)
    val time2 = System.nanoTime()
    val timeNeural = (time2-time1).toDouble/1000000000.0
    val (treeFolded, score, (timeLabeling, timeParsing, timeViterbi)) = parseFolded(scores, null)
    (unfoldTree(words, tags, treeFolded), (timeNeural, timeLabeling, timeParsing, timeViterbi))
  }

  private def parseFolded(scores: Array[Array[Array[Float]]], correctLabels:List[(Int, (Int, Int, Int, Int))]) : (ConstNode, Double, (Double, Double, Double)) = {
    val n = scores.length
    val rulesToUse = if(n>maxSentenceSizeForDicontinuousParsing) ParsingAlgorithm.rulesCFG else this.rulesToUse
    val (intLabelTree, (timeLabeling, timeParsing, timeViterbi), score) = ParsingAlgorithm.argmax(
        scoreTable                 = scores,
        coreLabelsDescs            = allS2I.coreNTdesc,
        correctLabels              = correctLabels,
        rulesToUse                 = rulesToUse,
        maxDiscElSize              = maxDiscElSize,
        maxGapSize                 = maxGapSize,
        withSpecialBinarizingLabel = !this.isExplicitBinarization
    )
    val treeFolded = Transforms.transformIntToStringLabels(intLabelTree, allS2I.n2i)
    assert(treeFolded.words.size == scores.length, s"wrong number of words in the parsed tree pred(${treeFolded.words.size}) != gold(${scores.length})")
    (treeFolded, score, (timeLabeling, timeParsing, timeViterbi))
  }

  private def eval(gold:ConstNode, pred:ConstNode) : (Double, Map[String, Double]) = {
    val goldStuff = gold.allNodes.filter(_.children.nonEmpty).map(node => (node.label, node.spans)).toSet
    val predStuff = pred.allNodes.filter(_.children.nonEmpty).map(node => (node.label, node.spans)).toSet
    val goldStuffSize  = goldStuff.size
    val predStuffSize  = predStuff.size
    val overlapSize = (goldStuff intersect predStuff).size
    val hammingPErr = predStuffSize-overlapSize
    val hammingRErr = goldStuffSize-overlapSize
    val hammingAErr = (hammingPErr+hammingRErr)/2.0
    val p = overlapSize.toDouble/predStuffSize
    val r = overlapSize.toDouble/goldStuffSize
    val f = FScore.f_score(p, r)

    (f, Map(
      "hamming_err_avg" -> hammingAErr,
      "hamming_err_p"   -> hammingPErr,
      "hamming_err_r"   -> hammingRErr,
      "correct"         -> overlapSize,
      "gold_size"       -> goldStuffSize,
      "pred_size"       -> predStuffSize,

      "correct_disc"    -> (goldStuff.filter(_._2.size > 1) intersect predStuff.filter(_._2.size > 1)).size,
      "gold_size_disc"  -> goldStuff.count(_._2.size > 1),
      "pred_size_disc"  -> predStuff.count(_._2.size > 1),

      "p"   -> p,
      "r"   -> r,
      "f"   -> f,
    ))
  }

  /**
    * called order 11
    * DOC validates
    */
  override def validate(devData: Iterable[IndexedInstance[ConstNode]]): (Double, Map[String, Double]) = {
    var goldSize    = 0.0
    var predSize    = 0.0
    var overlapSize = 0.0
    var disc_goldSize    = 0.0
    var disc_predSize    = 0.0
    var disc_overlapSize = 0.0
    for((IndexedInstance(_, gold), i) <- devData.zipWithIndex){
      DynetSetup.cg_renew()
      if(i%100==0)
        System.err.println(s"valid $i")
      val words = gold.leafsSorted.map(_.word)
      val tags  = gold.leafsSorted.map(_.label)
      val pred  = parse(words, tags)._1
      val res   = eval(gold, pred)
      overlapSize += res._2("correct"  )
      goldSize    += res._2("gold_size")
      predSize    += res._2("pred_size")
      disc_overlapSize += res._2("correct_disc"  )
      disc_goldSize    += res._2("gold_size_disc")
      disc_predSize    += res._2("pred_size_disc")
    }
    val p = overlapSize/predSize
    val r = overlapSize/goldSize
    val f = FScore.f_score(p, r)
    val disc_p = (disc_predSize, disc_goldSize) match {
      case (0.0, 0.0) => 1.0
      case (0.0,  _ ) => 0.0
      case ( _ ,  _ ) => disc_overlapSize/disc_predSize
    }
    val disc_r = if(disc_goldSize==0) 1 else disc_overlapSize/disc_goldSize
    val disc_f = FScore.f_score(disc_p, disc_r)
    (if(disc_goldSize==0) f else f+disc_f, Map(
      "p"   -> p,
      "r"   -> r,
      "f"   -> f,
      "disc_p"   -> disc_p,
      "disc_r"   -> disc_r,
      "disc_f"   -> disc_f,
    ))
  }

}
