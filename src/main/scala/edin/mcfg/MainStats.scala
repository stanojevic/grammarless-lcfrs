package edin.mcfg

import edin.constituency.representation.{ConstNode, ExportFormatParser}
import scala.collection.mutable.ArrayBuffer

object MainStats {

  def main(args:Array[String]) : Unit = {
    // val fn = "/home/milos/Projects/treebanks/originals_extracted/tiger_2.1/corpus/tiger_release_aug07.export"
//     val fn = "/home/milos/Projects/treebanks/transformed/tiger_2.2.export.0"
    val fn = "/home/milos/Projects/treebanks/transformed/negra.export.0"

    val trees = ConstNode.loadTreesFromFile(fn)

//    var c = 0
    val stats = trees.map(_.words.size).groupBy(identity).mapValues(_.size).toList
    println(stats.filter(_._1<=40).map(_._2).sum.toDouble/stats.map(_._2).sum)
    stats.sortBy(_._1).foreach{ case (i, c) => println(s"$i --> $c")}
//    for((tree, i) <- trees.zipWithIndex){
////      if(tree.allNodes.exists(n => n.children.size==1 && n.indices.size>=1 && n.spans != List((0, tree.leafs.size)))){
//      if(tree.allNodes.exists(n => n.spans.size>1)){
//        println(i)
//        c+/=1
//      }
//    }
//    println(c)
  }

  def main2(args:Array[String]) : Unit = {
    // val fn = "/home/milos/Projects/treebanks/originals_extracted/tiger_2.1/corpus/tiger_release_aug07.export"
    val fn = "/home/milos/Projects/treebanks/transformed/tiger_2.2.export.0"
    // val fn = "/home/milos/Projects/treebanks/transformed/negra.export.0"

    val trees = ConstNode.loadTreesFromFile(fn)

    val smallerSpans = ArrayBuffer[List[Int]]()
    val gapSizes     = ArrayBuffer[List[Int]]()
    val highSpans    = ArrayBuffer[ConstNode]()
    var totalSents   = 0

    for((tree, i) <- trees.zipWithIndex){
      totalSents += 1
      //      println(i)
      if(tree.allNodes.exists(_.spans.size>1)){
        val smallerSpan = tree.allNodes.filter(_.spans.size>1).flatMap{ node =>
          node.spans match {
            case List((a, b), (c, d)) => Some(math.min(b-a, d-c))
            case _                    => None
          }
        }
        val gaps = tree.allNodes.filter(_.spans.size>1).flatMap{ node =>
          node.spans match {
            case List((a, b), (c, d)) => Some(math.min(b-a, d-c))
            case _                    => None
          }
        }
        if(tree.allNodes.exists(_.spans.size>2))
          highSpans += tree
        if(smallerSpan.nonEmpty)
          smallerSpans += smallerSpan
        if(gaps.nonEmpty)
          gapSizes += gaps
      }
    }

    val totalDiscSizes =  gapSizes.size.toDouble
    var accDiscSizes   =  0d
    for((size, count) <- smallerSpans.map(_.max).groupBy(identity).mapValues(_.size).toList.sorted){
      accDiscSizes+=count
      println(s"disc $size  $count   ${100*accDiscSizes/totalDiscSizes}%")
    }
    println("\n"*3)
    val totalGapSizes =  gapSizes.size.toDouble
    var accGapSizes   =  0d
    for((size, count) <- gapSizes.map(_.max).groupBy(identity).mapValues(_.size).toList.sorted){
      accGapSizes+=count
      println(s"gap $size  $count   ${100*accGapSizes/totalGapSizes}%")
    }
    println("\n"*3)
    println("high order elems "+highSpans.size+" out of "+totalSents)
  }

}
