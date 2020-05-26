package edin.mcfg

import edin.constituency.representation.{ConstNode, ExportFormatParser}
import edin.general.Global
import edin.nn.DynetSetup

object MainIdealParse {

  case class CMDargs(
                      model_dir         : String =  null,
                      trees_file        : String =  null,
                      rules_to_use      : String =  null,
                      max_disc_sent_len : Int    =    50,
                    )

  /**
    * of course this can be optimized not to depend on any particular trained model but i'm lazy
    */

  def main(args: Array[String]) : Unit = {

    val parser = new scopt.OptionParser[CMDargs]("MCFL-2 parser") {
      head("MCFL-2 parser", "0.1")
      opt[ String   ]( "model-dir"                 ).action((x,c) => c.copy( model_dir                      = x )).required()
      opt[ String   ]( "rules-to-use"              ).action((x,c) => c.copy( rules_to_use                   = x )).required()
      opt[ String   ]( "trees-file"                ).action((x,c) => c.copy( trees_file                     = x )).required()
      opt[ Int      ]( "max-disc-sent-len"         ).action((x,c) => c.copy( max_disc_sent_len              = x ))
      help("help").text("prints this usage text")
    }

    parser.parse(args, CMDargs()) match {
      case Some(cmd_args) =>
        Global.printProcessId()
        DynetSetup.init_dynet()

        val model = new ParsingModel()
        model.loadFromModelDir(cmd_args.model_dir)
        model.setRulesToUse(cmd_args.rules_to_use)
        model.maxSentenceSizeForDicontinuousParsing = cmd_args.max_disc_sent_len

        System.err.println(s"loading trees START")
        val goldTrees = ConstNode.loadTreesFromFile(cmd_args.trees_file)
        System.err.println(s"loading trees END")
        for( (goldTree, i) <- goldTrees.zipWithIndex ){
          System.err.println(s"parsing ${i+1}/${goldTrees.size}")
          val tree = model.parseOracle(goldTree)
          println(ExportFormatParser.toString(tree, i))
        }
      case None =>
        System.err.println("You didn't specify all the required arguments")
        System.exit(-1)
    }
  }

}
