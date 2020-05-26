package edin.constituency

import java.io.PrintWriter

import edin.constituency.representation.ConstNode
import edin.mcfg.Transforms
import edin.supertagger.SuperTaggingModel

object MainTagUtils {

  case class CMDargs(
                      command           : String    = null,
                      inTrees           : String    = null,
                      inWords           : String    = null,
                      inTags            : String    = null,
                      outTrees          : String    = null,
                      outWords          : String    = null,
                      outTags           : String    = null,
                    )

  def main(args:Array[String]) : Unit = {
    val parser = new scopt.OptionParser[CMDargs]("MCFL-2 parser") {
      head("MCFL-2 parser", "0.1")
      opt[ String   ]( "command"     ).action((x,c) => c.copy( command    = x )).required()
      opt[ String   ]( "in-trees"    ).action((x,c) => c.copy( inTrees    = x ))
      opt[ String   ]( "in-words"    ).action((x,c) => c.copy( inWords    = x ))
      opt[ String   ]( "in-tags"     ).action((x,c) => c.copy( inTags     = x ))
      opt[ String   ]( "out-trees"   ).action((x,c) => c.copy( outTrees   = x ))
      opt[ String   ]( "out-words"   ).action((x,c) => c.copy( outWords   = x ))
      opt[ String   ]( "out-tags"    ).action((x,c) => c.copy( outTags    = x ))
      help("help").text("prints this usage text")
    }

    import ConstNode._

    parser.parse(args, CMDargs()) match {
      case Some(cmd_args) =>
        cmd_args.command match {
          case "renumber_export_trees" =>
            assert(cmd_args.inTrees  !=null)
            assert(cmd_args.outTrees !=null)
            assert(getFormat(cmd_args.inTrees ) == "export")
            assert(getFormat(cmd_args.outTrees) == "export")

            val trees = loadTreesFromFile(cmd_args.inTrees).toList
            saveTreesToFile(trees, cmd_args.outTrees)

          case "extract_tags_and_words" =>
            assert(cmd_args.inTrees !=null)
            assert(cmd_args.outWords!=null)
            assert(cmd_args.outTags !=null)
            val pwWord = new PrintWriter(cmd_args.outWords)
            val pwTag  = new PrintWriter(cmd_args.outTags )
            for(tree <- loadTreesFromFile(cmd_args.inTrees)){
              pwWord.println(tree.deleteEmptyNodes.leafsSorted.map(_.word ).mkString(" "))
              pwTag .println(tree.deleteEmptyNodes.leafsSorted.map(_.label).mkString(" "))
            }
            pwWord.close()
            pwTag.close()
          case "return_tags" =>
            assert(cmd_args.inTags   !=null)
            assert(cmd_args.inTrees  !=null)
            assert(cmd_args.outTrees !=null)
            val allTrees = loadTreesFromFile(cmd_args.inTrees).toList
            val allTags  = SuperTaggingModel.loadTokens(cmd_args.inTags).toList

//            val newTrees = (allTrees zip allTags).zipWithIndex.map{case ((tree, tags), i) => System.err.println(s"tree $i : ${tree.words.mkString}") ;Transforms.replaceTags(tags, tree)}
            val newTrees = (allTrees zip allTags).map{case (tree, tags) => Transforms.replaceTags(tags, tree)}
            saveTreesToFile(newTrees, cmd_args.outTrees)
        }

      case None =>
        System.err.println("You didn't specify all the required arguments")
        System.exit(-1)
    }
  }


}
