package edin.mcfg

import edin.constituency.representation.ConstNode
import edin.nn.model.Any2Int
import edin.general.RichScala._

object Transforms {

  // collapses all unaries and replaces POS tags with a collapsed preterminal unary or special END
  def transformFold(tree:ConstNode) : ConstNode = tree match {
    case ConstNode("ROOT", List(child)) =>
      // unary root
      transformFold(child) // deleting unary ROOT nodes
    case ConstNode(label, List(child))  =>
      // unary
      if(child.children.nonEmpty){
        transformFold(ConstNode(label+"_"+child.label, child.children))
      }else{
        child.copy(label=label+"_"+"END")
      }
    case ConstNode(label, Nil) =>
      // terminal
      tree.copy(label="END")
    case ConstNode(label, children) =>
      // n-ary
      tree.copy(children=children map transformFold)
  }

  def replaceWordsInplace(words:List[String], node:ConstNode) : ConstNode = {
    for((leaf, word) <- node.leafsSorted zip words)
      leaf.setTerminalInfo(word, leaf.indices.head)
    node
  }

  def replaceTags(tags:List[String], node:ConstNode) : ConstNode = node match {
    case ConstNode(_, Nil     ) => node.copy(tags(node.indices.head))
    case ConstNode(_, children) => node.copy(children = children.map(replaceTags(tags, _)))
  }

  def transformUnfold(words:List[String], tags:List[String], node:ConstNode) : ConstNode =
    transformUnfold(node) |> (replaceTags(tags, _)) |> (replaceWordsInplace(words, _)) |> (_.deBinarized)


  def transformUnfold(node:ConstNode) : ConstNode = transformUnfoldRec(node).addDummyRootNode
  private def transformUnfoldRec(node:ConstNode) : ConstNode = {
      val newChildren = node.children.map(transformUnfoldRec)
      val sublabels = node.label.split("_").toList
      val bottom = node.copy(label=sublabels.last, children=newChildren)
      sublabels.init.foldRight(bottom){ (label, bottom) => bottom.copy(label=label, children=List(bottom)) }
  }

  def transformIntToStringLabels(node:ConstNode, n2i:Any2Int[String]) : ConstNode =
    node.copy(
      label    = n2i(node.label.toInt),
      children = node.children.map(transformIntToStringLabels(_, n2i))
    )

}
