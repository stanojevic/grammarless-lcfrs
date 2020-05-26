package edin.constituency.representation.headrules

import edin.constituency.representation.ConstNode
import edin.algorithms.AutomaticResourceClosing.linesFromFile
import edin.general.Global

// TODO add rules for other languages:

// TODO      CHINESE

// TODO option 0 -- Chinese
// TODO Appendix of "Shallow Semantic Parsing of Chinese" by Sun and Jurafsky
// TODO https://www.aclweb.org/anthology/N04-1032.pdf

// TODO option 1 -- many other Chinese head finders
// TODO https://github.com/stanfordnlp/CoreNLP/tree/master/src/edu/stanford/nlp/trees/international/pennchinese
// TODO contains Bikkel thesis (2004), Levy and Manning (2003), Sun and Jurafsky (2004),

// TODO      SPMRL

// TODO option 0 -- use some simple rules (that don't cover all SPMRL languages)
// TODO https://www.cs.cmu.edu/~afm/Home_files/acl2015_reduction.pdf
// TODO "Parsing as Reduction" by Fernandez-Gonzalez and Martins
// TODO Section 5.2  Results on the SPMRL Datasets

// TODO option 1 -- ???
// TODO "Incorporating Semi-supervised Features into Discontinuous Easy-firstConstituent Parsing" by Versley
// TODO http://pauillac.inria.fr/~seddah/proc_st2014/Versley_31_Paper.pdf

// TODO option 2 -- use provided parallel SPMRL dependency treebank
// TODO http://pauillac.inria.fr/~seddah/proc_st2014/Alpage_33_Paper.pdf
// TODO "Multilingual Discriminative Shift-Reduce Phrase Structure Parsing forthe SPMRL 2014 Shared Task" by Crabbe and Seddah
// TODO Section 3.1  Head assignation procedure

// TODO option 3 -- alignment with the parallel dependency treebank alternative
// TODO "Multilingual discriminative lexicalized phrase structure parsing" by Crabbe

class CollinsRules(rulesFile:String, headFinal:Boolean = true) {

  private val rules = {
    val r = collection.mutable.Map[String, List[(Boolean, Set[String])]]()
    linesFromFile(Global.projectDir+"/src/main/scala/edin/constituency/representation/headrules/"+rulesFile+".headrules").
      filterNot(_ matches """^\%.*""").
      filterNot(_ matches """^\s*$""").
      map(_.toLowerCase split " +").
      foreach{ fields =>
        val parent = fields(0)
        val isLeftToRight = (fields(1) : @unchecked) match {
          case "left-to-right" => true
          case "right-to-left" => false
        }
        val options = fields.tail.tail.toSet
        r(parent) = r.getOrElse(parent, Nil) :+ (isLeftToRight, options)
      }
    r
  }

  def chooseHead(tree:ConstNode) : Int = chooseHead(tree.label, tree.children.map(_.label))
  def chooseHead(parent:String, children:List[String]) : Int = chooseHeadLowercased(parent.toLowerCase, children.map(_.toLowerCase))

  private def chooseHeadLowercased(parent:String, children:List[String]) : Int =
    rules.get(parent) match {
      case None =>
        if(headFinal) children.size-1 else 0
      case Some(rules) =>
        rules.find{case (_, options) => children.exists(options)} match {
          case None =>
            if(headFinal) children.size-1 else 0
          case Some((true, options)) =>
            children.takeWhile(child => !options(child)).size
          case Some((false, options)) =>
            children.size - children.reverse.takeWhile(child => !options(child)).size - 1
        }
    }

  def assignHeadsToTree(tree:ConstNode) : Unit = tree.allNodesPostorder.foreach{
    case node@ConstNode(_, Nil) =>
      node.headChildIndex    = 0
      node.headTerminalIndex = node.indices.head
    case node@ConstNode(_, children) =>
      node.headChildIndex = chooseHead(node)
      node.headTerminalIndex = children(node.headChildIndex).headTerminalIndex
  }

}
