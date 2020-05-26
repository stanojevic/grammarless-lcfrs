package edin.constituency.grammar

import edin.constituency.chart.Query.Query
import edin.constituency.grammar.GrammarCFG.{CFGBinary, CFGUnary, CFGitem}

object GrammarCFG{

  type NonTerm = Int

  sealed case class CFGitem(i:Int, j:Int, binary:Boolean, label:NonTerm) extends Item[CFGitem] {

    override def signature: List[Int] = List(i, j, label)

    override def equals(o: Any): Boolean = o match {
      case CFGitem(`i`, `j`, `binary`, `label`) => true
      case _ => false
    }

    override var bestScore: Double = _
    override var priority: Double = _

  }

  sealed case class CFGUnary(parent:NonTerm, child:NonTerm)  extends UnaryRule [CFGitem] {

    override def inferences(itemA:CFGitem) : Stream[CFGitem] =
      Stream(CFGitem(itemA.i, itemA.j, true, parent))

  }

  sealed case class CFGBinary(parent:NonTerm, left:NonTerm, right:NonTerm, triggerIsLeft:Boolean) extends BinaryRule[CFGitem] {

    override def inferences(itemA:CFGitem, itemC:CFGitem) : Stream[CFGitem] =
      if(triggerIsLeft)
        Stream(CFGitem(itemA.i, itemC.j, true, parent))
      else
        Stream(CFGitem(itemC.i, itemA.j, true, parent))

    override def queryVectors(itemA: CFGitem): Seq[Query] =
      if(triggerIsLeft){
        assert(left == itemA.label)
        List(List(itemA.j, -1, right))
      }else{
        assert(right == itemA.label)
        List(List(-1, itemA.i, left))
      }
  }

}

class GrammarCFG extends Grammar[CFGitem, CFGUnary, CFGBinary] {

  override def initParsingBatch(sents: List[List[String]]): GrammarState[CFGitem, CFGUnary, CFGBinary] = ???

}

