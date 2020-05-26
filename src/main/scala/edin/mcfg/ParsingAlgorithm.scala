package edin.mcfg

import edin.constituency.representation.ConstNode

import scala.reflect.ClassTag
import spire.syntax.cfor.cforRange
import java.lang.Math.{max, min}

import scala.collection.immutable.BitSet

object ParsingAlgorithm {

  def main(args:Array[String]) : Unit = {
    val scoreTable =
      Array(
        Array(
          Array[Float]( 0, 1, 0, 0, 0, 0),  // (0,0)
          Array[Float]( 0, 0, 1, 0, 0, 0),  // (0,1)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (0,2)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (0,3)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (0,4)
        ),
        Array(
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (1,0)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (1,1)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (1,2)
          Array[Float]( 0, 0, 0, 1, 0, 0),  // (1,3)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (1,4)
        ),
        Array(
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (2,0)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (2,1)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (2,2)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (2,3)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (2,4)
        ),
        Array(
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (3,0)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (3,1)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (3,2)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (3,3)
          Array[Float]( 0, 0, 0, 0, 1, 0),  // (3,4)
        ),
        Array(
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (4,0)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (4,1)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (4,2)
          Array[Float]( 0, 0, 0, 0, 0, 0),  // (4,3)
          Array[Float]( 0, 0, 0, 0, 0, 1),  // (4,4)
        ),
      )
    val coreLabelsDescs = Array(
      Array(0, 1, 2, 3),   /// 0-1 3        1-2  2      2-3   3
    )
    val maxDiscElSize = 5
    val maxGapSize = 1
    val (tree, (timeLabeling, timeParsing, timeViterbi), score) = argmax(scoreTable, coreLabelsDescs, null, rulesMCFG, maxDiscElSize=maxDiscElSize, maxGapSize=maxGapSize)
    tree.visualize()
    println((timeLabeling, timeParsing, timeViterbi))
  }

  val rulesCFG       = BitSet(3)            // O( L*n^2 + n^3 )
  val rulesWNnonRare = BitSet(3 to 9:_*)    // O( L*n^4 + n^5 )
  val rulesWN        = rulesWNnonRare + 11  // O( L*n^4 + n^6 )
  val rulesMCFG      = BitSet(3 to 14:_*)   // O( L*n^4 + n^6 )

  def argmax(
              scoreTable                 : ScoreTable,
              coreLabelsDescs            : CoreLabelsDescs,
              correctLabels              : List[(Label, Span)],
              rulesToUse                 : BitSet,
              maxDiscElSize              : Int = 150,
              maxGapSize                 : Int = 150,
              withSpecialBinarizingLabel : Boolean = true,
            ) : (ConstNode, (Double, Double, Double), Double) = { // (currentSpan, currentLabel, child1Span, child2Span)
    assert(maxDiscElSize<=1000)
    assert(maxGapSize<=1000)
    val n = scoreTable.length
    val time1 = System.nanoTime()
    val (optimalLabelFanOut1Id, optimalLabelFanOut1Score, optimalLabelFanOut2Id, optimalLabelFanOut2Score) = findOptimalLabels(scoreTable, coreLabelsDescs, correctLabels              , rulesToUse, maxDiscElSize=maxDiscElSize, maxGapSize=maxGapSize, withSpecialBinarizingLabel=withSpecialBinarizingLabel)
    val time2 = System.nanoTime()
    val (       chartFanOut1Id,        chartFanOut1Score,        chartFanOut2Id,        chartFanOut2Score) = findOptimalBackPointers(optimalLabelFanOut1Score, optimalLabelFanOut2Score, rulesToUse, maxDiscElSize=maxDiscElSize, maxGapSize=maxGapSize)
    val time3 = System.nanoTime()
    val (tree, score) = backTrackViterbi((0, n, -1, -1))(optimalLabelFanOut1Id, optimalLabelFanOut1Score, optimalLabelFanOut2Id, optimalLabelFanOut2Score, chartFanOut1Id, chartFanOut2Id)
    val time4 = System.nanoTime()
    val timeInfo = ((time2-time1).toDouble/1000000000.0, (time3-time2).toDouble/1000000000.0, (time4-time3).toDouble/1000000000.0)
    (tree, timeInfo, score)
  }

          type BackPointer              = Array[Int] // Array(a1, b1, c1, d1, a2, b2, c2, d2, ruleId) // if proj then cx&dx < 0
          type Label                    = Int
          type ScoreTable               = Array[Array[Array[Float]]]
          type CoreLabelsDescs          = Array[Array[Int]] // Array(coreLabelId, beforeFakeLabelId, leftCompFakeLabelId, gapCompFakeLabelId, rightCompFakeLabelId, afterFakeLabelId) // if proj then use coreLabelId
          type Span                     = (Int, Int, Int, Int)

  private type OptimalLabelFanOut1Id    = Array[Array[Label]]
  private type OptimalLabelFanOut2Id    = Array[Array[Array[Array[Label]]]]
  private type OptimalLabelFanOut1Score = Array[Array[Float]]
  private type OptimalLabelFanOut2Score = Array[Array[Array[Array[Float]]]]
  private type OptimalLabelResult       = (OptimalLabelFanOut1Id, OptimalLabelFanOut1Score, OptimalLabelFanOut2Id, OptimalLabelFanOut2Score)

  private type ChartFanOut1Id           = Array[Array[BackPointer]]
  private type ChartFanOut2Id           = Array[Array[Array[Array[BackPointer]]]]
  private type ChartFanOut1Score        = Array[Array[Float]]
  private type ChartFanOut2Score        = Array[Array[Array[Array[Float]]]]
  private type ChartResult              = (ChartFanOut1Id, ChartFanOut1Score, ChartFanOut2Id, ChartFanOut2Score)

  // O(n)
  private def backTrackViterbi(
                                span                     : (Int, Int, Int, Int)
                              )(implicit
                                optimalLabelFanOut1Id    : OptimalLabelFanOut1Id,
                                optimalLabelFanOut1Score : OptimalLabelFanOut1Score,
                                optimalLabelFanOut2Id    : OptimalLabelFanOut2Id,
                                optimalLabelFanOut2Score : OptimalLabelFanOut2Score,
                                chartFanOut1Id           : ChartFanOut1Id,
                                chartFanOut2Id           : ChartFanOut2Id,
                              ) : (ConstNode, Double) = {
    val (a, b, c,  d) = span
    val (bp, label, labelScore) = if(c < 0) {
      // projective node
      assert(d < 0)
      (chartFanOut1Id(a)(b), optimalLabelFanOut1Id(a)(b), optimalLabelFanOut1Score(a)(b))
    }else {
      // non-projective node
      (chartFanOut2Id(a)(b)(c)(d), optimalLabelFanOut2Id(a)(b)(c)(d), optimalLabelFanOut2Score(a)(b)(c)(d))
    }

    if(span._1+1 == span._2 && span._3<0){
      // preterminal node ; end recursion
      val node = ConstNode(s"$label", Nil)
      node.setTerminalInfo(s"word_${span._1}", span._1)
      (node, labelScore)
    }else{
      val firstSpan  = (bp(0), bp(1), bp(2), bp(3))
      val secondSpan = (bp(4), bp(5), bp(6), bp(7))
      val (leftChild , lScore) = backTrackViterbi(firstSpan)
      val (rightChild, rScore) = backTrackViterbi(secondSpan)

      var flatChildren = List[ConstNode]()

      if(leftChild.label=="-1")
        flatChildren = leftChild.children
      else
        flatChildren = List(leftChild)

      if(rightChild.label=="-1")
        flatChildren ++= rightChild.children
      else
        flatChildren :+= rightChild

      val res = ConstNode(
        s"$label",
        flatChildren
      )
      (res, labelScore+lScore+rScore)
    }
  }


  // O(n^6)
  private def findOptimalBackPointers(
                                       optimalLabelFanOut1Score : OptimalLabelFanOut1Score,
                                       optimalLabelFanOut2Score : OptimalLabelFanOut2Score,
                                       rulesToUse               : BitSet,
                                       maxDiscElSize            : Int,
                                       maxGapSize               : Int,
                                     ) : ChartResult = {
    assert(maxDiscElSize<=1000)
    val n = optimalLabelFanOut1Score.length-1
    val chartFanOut1Id    = emptyTableFanOut1[BackPointer](n+1)
    val chartFanOut1Score = emptyTableFanOut1[Float](n+1)
    val chartFanOut2Id    = if(rulesToUse == BitSet(3)) null else emptyTableFanOut2[BackPointer](n+1)
    val chartFanOut2Score = if(rulesToUse == BitSet(3)) null else emptyTableFanOut2[Float](n+1)

    assert(rulesToUse.max <=14)
    assert(!(rulesToUse contains 1), "unary rule is never used")
    assert(!(rulesToUse contains 2), "unary rule is never used")
    val useRule3  = rulesToUse contains 3
    val useRule4  = rulesToUse contains 4
    val useRule5  = rulesToUse contains 5
    val useRule6  = rulesToUse contains 6
    val useRule7  = rulesToUse contains 7
    val useRule8  = rulesToUse contains 8
    val useRule9  = rulesToUse contains 9
    val useRule10 = rulesToUse contains 10
    val useRule11 = rulesToUse contains 11
    val useRule12 = rulesToUse contains 12
    val useRule13 = rulesToUse contains 13
    val useRule14 = rulesToUse contains 14


    // preterminals
    cforRange(0 until n){ a =>
      chartFanOut1Score(a)(a+1) = optimalLabelFanOut1Score(a)(a+1)
    }

    val minGapSize = 1

    cforRange(2 to n){ spanSize =>
      // projective optimization
      cforRange(0 to n-spanSize){ a =>
        val b = a+spanSize

        var bestBP    : BackPointer = null
        var bestScore : Float       = Float.NegativeInfinity

        // Maier rule 3  // concat 2 components                  CFG   O(n^3)
        if(useRule3){
          cforRange(a+1 until b){ s1 =>
            val score = chartFanOut1Score(a)(s1)+chartFanOut1Score(s1)(b)
            if(score > bestScore){
              bestScore = score
              bestBP = Array(a, s1, -1, -1, s1, b, -1, -1, 3)
            }
          }
        }
        // Maier rule 5  // concat 3 components     well-nested MCFG   O(n^4)
        if(useRule5){
          cforRange(a+1 to b-2){ s1 =>
            val s2Start = if(s1-a>maxDiscElSize && b-maxDiscElSize>=s1+1) b-maxDiscElSize else s1+1
            cforRange(s2Start to b-1){ s2 =>
              val score = chartFanOut2Score(a)(s1)(s2)(b)+chartFanOut1Score(s1)(s2)
              if(score > bestScore){
                bestScore = score
                bestBP = Array(a, s1, s2, b, s1, s2, -1, -1, 5)
              }
            }
          }
        }
        // Maier rule 14 // concat 4 components NOT well-nested MCFG   O(n^5)
        if(useRule14){
          cforRange(a+1 until b-2){ s1 =>
            cforRange(s1+1 until b-1){ s2 =>
              var s3Start = s2+1
              var s3End   = b-0
              if(s1-a>maxDiscElSize && s2-s1>maxDiscElSize && b-2>maxDiscElSize+maxDiscElSize){
                s3Start = Int.MaxValue
                s3End   = 0
              }else if(s1-a>maxDiscElSize && s2-s1>maxDiscElSize){
                s3Start = max(b-maxDiscElSize, s2+1)
                s3End   = min(s2+maxDiscElSize, b)
              }else if(s1-a>maxDiscElSize){
                s3End   = min(s2+maxDiscElSize, b)
              }else if(s2-s1>maxDiscElSize){
                s3Start = max(b-maxDiscElSize, s2+1)
              }else{
                s3Start = s2+1
                s3End   = b-0
              }
              cforRange(s3Start until s3End){ s3 =>
                val score = chartFanOut2Score(a)(s1)(s2)(s3) + chartFanOut2Score(s1)(s2)(s3)(b)
                if (score > bestScore) {
                  bestScore = score
                  bestBP = Array(a, s1, s2, s3, s1, s2, s3, b, 14)
                }
              }
            }
          }
        }

        chartFanOut1Id(a)(b)    = bestBP
        chartFanOut1Score(a)(b) = bestScore + optimalLabelFanOut1Score(a)(b)
      }

      // non-projective optimization
      if(useRule4 || useRule6 || useRule7 || useRule8 || useRule9 || useRule10 || useRule11 || useRule12 || useRule13){
        cforRange(1 to spanSize-1){ leftSize =>
          val rightSize = spanSize-leftSize
          if(leftSize<maxDiscElSize || rightSize<maxDiscElSize) {
            cforRange(0 to n - spanSize - minGapSize) { a =>
              val b = a + leftSize
              val cEnd = min(b+maxGapSize, n-rightSize)
              cforRange(b + minGapSize to cEnd) { c =>
                val d = c + rightSize

                var bestBP: BackPointer = null
                var bestScore: Float = Float.NegativeInfinity

                // Maier rule 4   O(n^4)
                if (useRule4) {
                  val score = chartFanOut1Score(a)(b) + chartFanOut1Score(c)(d)
                  if (score > bestScore) {
                    bestScore = score
                    bestBP = Array(a, b, -1, -1, c, d, -1, -1, 4)
                  }
                }
                // Maier rule 6   O(n^5)
                if (useRule6) {
                  cforRange(c + 1 until d) { s1 =>
                    val score = chartFanOut2Score(a)(b)(c)(s1) + chartFanOut1Score(s1)(d)
                    if (score > bestScore) {
                      bestScore = score
                      bestBP = Array(a, b, c, s1, s1, d, -1, -1, 6)
                    }
                  }
                }
                // Maier rule 7   O(n^5)
                if (useRule7) {
                  cforRange(c + 1 until d) { s1 =>
                    val score = chartFanOut2Score(a)(b)(s1)(d) + chartFanOut1Score(c)(s1)
                    if (score > bestScore) {
                      bestScore = score
                      bestBP = Array(a, b, s1, d, c, s1, -1, -1, 7)
                    }
                  }
                }
                // Maier rule 8   O(n^5)
                if (useRule8) {
                  cforRange(a + 1 until b) { s1 =>
                    val score = chartFanOut2Score(a)(s1)(c)(d) + chartFanOut1Score(s1)(b)
                    if (score > bestScore) {
                      bestScore = score
                      bestBP = Array(a, s1, c, d, s1, b, -1, -1, 8)
                    }
                  }
                }
                // Maier rule 9   O(n^5)
                if (useRule9) {
                  cforRange(a + 1 until b) { s1 =>
                    val score = chartFanOut1Score(a)(s1) + chartFanOut2Score(s1)(b)(c)(d)
                    if (score > bestScore) {
                      bestScore = score
                      bestBP = Array(a, s1, -1, -1, s1, b, c, d, 9)
                    }
                  }
                }
                // Maier rule 10   /// not well-nested  and appears 0 times in English and German  O(n^6)
                if (useRule10) {
                  // can be further restricted with maxDiscElSize
                  cforRange(a + 1 until b) { s1 =>
                    cforRange(c + 1 until d) { s2 =>
                      val score = chartFanOut2Score(a)(s1)(c)(s2) + chartFanOut2Score(s1)(b)(s2)(d)
                      if (score > bestScore) {
                        bestScore = score
                        bestBP = Array(a, s1, c, s2, s1, b, s2, d, 10)
                      }
                    }
                  }
                }
                // Maier rule 11   ///  is well-nested but appears 0 times in German  O(n^6)
                if (useRule11) {
                  // can be further restricted with maxDiscElSize
                  cforRange(a + 1 until b) { s1 =>
                    cforRange(c + 1 until d) { s2 =>
                      val score = chartFanOut2Score(a)(s1)(s2)(d) + chartFanOut2Score(s1)(b)(c)(s2)
                      if (score > bestScore) {
                        bestScore = score
                        bestBP = Array(a, s1, s2, d, s1, b, c, s2, 11)
                      }
                    }
                  }
                }
                // Maier rule 12   /// not well-nested   O(n^6)
                if (useRule12) {
                  // can be further restricted with maxDiscElSize
                  cforRange(c + 1 until d - 1) { s1 =>
                    cforRange(s1 + 1 until d) { s2 =>
                      val score = chartFanOut2Score(a)(b)(s1)(s2) + chartFanOut2Score(c)(s1)(s2)(d)
                      if (score > bestScore) {
                        bestScore = score
                        bestBP = Array(a, b, s1, s2, c, s1, s2, d, 12)
                      }
                    }
                  }
                }
                // Maier rule 13   /// not well-nested   O(n^6)
                if (useRule13) {
                  // can be further restricted with maxDiscElSize
                  cforRange(a + 1 until b - 1) { s1 =>
                    cforRange(s1 + 1 until b) { s2 =>
                      val score = chartFanOut2Score(a)(s1)(s2)(b) + chartFanOut2Score(s1)(s2)(c)(d)
                      if (score > bestScore) {
                        bestScore = score
                        bestBP = Array(a, s1, s2, b, s1, s2, c, d, 13)
                      }
                    }
                  }
                }

                if (chartFanOut2Id != null) {
                  chartFanOut2Id(a)(b)(c)(d) = bestBP
                  chartFanOut2Score(a)(b)(c)(d) = bestScore + optimalLabelFanOut2Score(a)(b)(c)(d)

//                  // DEBUGGING PART
//                  val bp = bestBP
//                  val firstSpan  = (bp(0), bp(1), bp(2), bp(3))
//                  val secondSpan = (bp(4), bp(5), bp(6), bp(7))
//                  for(span <- List(firstSpan, secondSpan)){
//                    if(span._3<0 && span._2-span._1!=1 && chartFanOut1Id(span._1)(span._2)==null || span._3>=0 && chartFanOut2Id(span._1)(span._2)(span._3)(span._4)==null){
//                      System.err.println("debugging ")
//                    }
//                  }

                }
              }
            }
          }
        }
      }
    }

    (chartFanOut1Id, chartFanOut1Score, chartFanOut2Id, chartFanOut2Score)
  }

  // O( L*n^2 + L*n^4 + L*n )
  private def findOptimalLabels(
                                 scoreTable                 : ScoreTable,
                                 coreLabelsDescs            : CoreLabelsDescs,
                                 correctLabels              : List[(Label, Span)],
                                 rulesToUse                 : BitSet,
                                 maxDiscElSize              : Int,
                                 maxGapSize                 : Int,
                                 withSpecialBinarizingLabel : Boolean
                               ) : OptimalLabelResult = {
    val n = scoreTable.length
    val useOnlyCFG = rulesToUse == BitSet(3)
    val optimalLabelFanOut1Id    = emptyTableFanOut1[Label](n) // n actually means array of n+1
    val optimalLabelFanOut1Score = emptyTableFanOut1[Float](n)
    val optimalLabelFanOut2Id    = if(useOnlyCFG) null else emptyTableFanOut2[Label](n)
    val optimalLabelFanOut2Score = if(useOnlyCFG) null else emptyTableFanOut2[Float](n)

    val costAugmentedDecodingCost = if(correctLabels == null) 0f else 1f

    cforRange(1 to n){spanSize =>
      // O(L*n^2)
      // projective optimization
      cforRange(0 to n-spanSize){ a =>
        val b = a+spanSize

        var bestLabel : Label = -1
        var bestScore : Float = Float.NegativeInfinity

        cforRange(0 until coreLabelsDescs.length){ i =>
          val labelDesc = coreLabelsDescs(i)
          val labelCore   = labelDesc(0)
          val score = scoreTable(a)(b-1)(labelCore) + costAugmentedDecodingCost
          if(score > bestScore){
            bestScore = score
            bestLabel = labelCore
          }
        }
        if(withSpecialBinarizingLabel && bestScore<0f && !(a==0 && b==n) && spanSize>1){
          optimalLabelFanOut1Id(a)(b)    = -1 // special binarization label
          optimalLabelFanOut1Score(a)(b) = 0f
        }else{
          optimalLabelFanOut1Id(a)(b)    = bestLabel
          optimalLabelFanOut1Score(a)(b) = bestScore
        }
      }

      // O(L*n^4)
      // non-projective optimization
      if( ! useOnlyCFG){
        cforRange(1 to spanSize-1){ leftSize =>
          val rightSize = spanSize-leftSize
          if(leftSize<maxDiscElSize || rightSize<maxDiscElSize){
            cforRange(0 until n-spanSize){ a =>
              val b = a+leftSize
              val cEnd = min(b+maxGapSize, n-rightSize)
              cforRange(b+1 to cEnd){ c =>
                val d = c+rightSize

                var bestLabel : Label = -1
                var bestScore : Float = Float.NegativeInfinity

                cforRange(0 until coreLabelsDescs.length){ i =>
                  val labelDesc = coreLabelsDescs(i)
                  val coreLabel   = labelDesc(0)
                  val labelLeft   = labelDesc(1)
                  val labelGap    = labelDesc(2)
                  val labelRight  = labelDesc(3)
                  val score =
                    scoreTable(    a        )(     b-1     )( labelLeft   ) +
                    scoreTable(    b        )(     c-1     )( labelGap    ) +
                    scoreTable(    c        )(     d-1     )( labelRight  ) +
                    costAugmentedDecodingCost
                  if(score > bestScore){
                    bestScore = score
                    bestLabel = coreLabel
                  }
                }
                if(withSpecialBinarizingLabel && bestScore<0f){
                  optimalLabelFanOut2Id(a)(b)(c)(d)    = -1 // special binarization label
                  optimalLabelFanOut2Score(a)(b)(c)(d) = 0f
                }else{
                  optimalLabelFanOut2Id(a)(b)(c)(d)    = bestLabel
                  optimalLabelFanOut2Score(a)(b)(c)(d) = bestScore
                }
              }
            }
          }
        }
      }
    }

    // O(L*n)
    // cost augmented decoding
    if(correctLabels != null){
      // cost augmented decoding
      for((correctLabel, (a, b, c, d)) <- correctLabels){
        if(c < 0){
          // projective
          var bestLabel : Label = -1
          var bestScore : Float = if(b-a==1 || b-a==n) Float.NegativeInfinity else if(-1==correctLabel) 0f else 1f

          cforRange(0 until coreLabelsDescs.length){ i =>
            val costAugmentedDecodingCost = if(i==correctLabel) 0f else 1f

            val labelDesc = coreLabelsDescs(i)
            val labelCore = labelDesc(0)
            val score = scoreTable(a)(b-1)(labelCore) + costAugmentedDecodingCost
            if(score > bestScore){
              bestScore = score
              bestLabel = labelCore
            }
          }
          optimalLabelFanOut1Id(a)(b)    = bestLabel
          optimalLabelFanOut1Score(a)(b) = bestScore
        }else if(! useOnlyCFG){
          // non-projective
          var bestLabel : Label = -1
          var bestScore : Float = if(-1==correctLabel) 0f else 1f

          cforRange(0 until coreLabelsDescs.length){ i =>
            val costAugmentedDecodingCost = if(i==correctLabel) 0f else 1f
            val labelDesc = coreLabelsDescs(i)
            val coreLabel   = labelDesc(0)
            val labelLeft   = labelDesc(1)
            val labelGap    = labelDesc(2)
            val labelRight  = labelDesc(3)
            val score = scoreTable( a )( b-1 )( labelLeft   ) +
                        scoreTable( b )( c-1 )( labelGap    ) +
                        scoreTable( c )( d-1 )( labelRight  ) +
                        costAugmentedDecodingCost
            if(score > bestScore){
              bestScore = score
              bestLabel = coreLabel
            }
          }
          optimalLabelFanOut2Id(a)(b)(c)(d)    = bestLabel
          optimalLabelFanOut2Score(a)(b)(c)(d) = bestScore
        }
      }
    }

    (optimalLabelFanOut1Id, optimalLabelFanOut1Score, optimalLabelFanOut2Id, optimalLabelFanOut2Score)
  }

  private def emptyTableFanOut1[@specialized(Int, Float) A](n:Int)(implicit classTag: ClassTag[A]) : Array[Array[A]] = {
    val c1 = new Array[Array[A]](n+1)
    cforRange(0 to n){ a =>
      c1(a) = new Array(n+1)
    }
    c1
  }

  private def emptyTableFanOut2[@specialized(Int, Float) A](n:Int)(implicit classTag: ClassTag[A]) : Array[Array[Array[Array[A]]]] = {
    val c1 = emptyTableFanOut1[Array[Array[A]]](n)
    cforRange(0 to n){ a =>
      cforRange(0 to n){ b =>
        c1(a)(b) = emptyTableFanOut1(n)
      }
    }
    c1
  }

}
