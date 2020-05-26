package edin.mcfg

import java.io.PrintWriter

import edin.constituency.representation.{ConstNode, ExportFormatParser}
import edin.general.Global
import edin.nn.DynetSetup
import edin.supertagger.SuperTaggingModel

object MainParse {

  case class CMDargs(
                      model_dir             : String  =  null,
                      words_file            : String  =  null,
                      rules_to_use          : String  =  null,
                      tags_file             : String  =  null,
                      output_file           : String  =  null,
                      span_breakage_scoring : Boolean = false,
                      store_speed_stats     : Boolean = false,
                      max_disc_sent_len     : Int     =    50,
                    )

  def main(args: Array[String]) : Unit = {

    val parser = new scopt.OptionParser[CMDargs]("MCFL-2 parser") {
      head("MCFL-2 parser", "0.1")
      opt[ String   ]( "model-dir"                 ).action((x,c) => c.copy( model_dir                      = x )).required()
      opt[ String   ]( "words-file"                ).action((x,c) => c.copy( words_file                     = x )).required()
      opt[ String   ]( "rules-to-use"              ).action((x,c) => c.copy( rules_to_use                   = x )).required()
      opt[ String   ]( "tags-file"                 ).action((x,c) => c.copy( tags_file                      = x ))
      opt[ Boolean  ]( "span-breakage-scoring"     ).action((x,c) => c.copy( span_breakage_scoring          = x ))
      opt[ Boolean  ]( "store-speed-stats"         ).action((x,c) => c.copy( store_speed_stats              = x ))
      opt[ Int      ]( "max-disc-sent-len"         ).action((x,c) => c.copy( max_disc_sent_len              = x ))
      opt[ String   ]( "output-file"               ).action((x,c) => c.copy( output_file                    = x )).required()
      help("help").text("prints this usage text")
    }

    parser.parse(args, CMDargs()) match {
      case Some(cmd_args) =>
        Global.printProcessId()
        DynetSetup.init_dynet()

        System.err.println("parsing started at "+Global.currentTimeHumanFormat)

        val model = new ParsingModel()
        model.loadFromModelDir(cmd_args.model_dir)
        model.setRulesToUse(cmd_args.rules_to_use)
        model.spanBreakageScoring = cmd_args.span_breakage_scoring
        model.maxSentenceSizeForDicontinuousParsing = cmd_args.max_disc_sent_len

        System.err.println(s"loading words START")
        val allSents = SuperTaggingModel.loadTokens(cmd_args.words_file).toList
        System.err.println(s"loading words END")

        val outPw = new PrintWriter(cmd_args.output_file)

        val statsPw = if(cmd_args.store_speed_stats){
          new PrintWriter(cmd_args.output_file+".time_stats")
        }else{
          null
        }
        if(statsPw!=null)
          statsPw.println("len,timeNeural,timeLabeling,timeParsing,timeViterbi")

        var (totalTimedSents, totalTimeNeural, totalTimeLabeling, totalTimeParsing, totalTimeViterbi) = (0, 0.0, 0.0, 0.0, 0.0)

        def processParseResult(i:Int, input: => (ConstNode, (Double, Double, Double, Double))) : Unit = {
          System.err.println(s"parsing ${i+1}/${allSents.size}")
          val (tree, (timeNeural, timeLabeling, timeParsing, timeViterbi)) = input
          if(statsPw!=null && tree.words.size <= model.maxSentenceSizeForDicontinuousParsing){
            statsPw.println(s"${tree.words.size},$timeNeural,$timeLabeling,$timeParsing,$timeViterbi")
            totalTimedSents   += 1
            totalTimeNeural   += timeNeural
            totalTimeLabeling += timeLabeling
            totalTimeParsing  += timeParsing
            totalTimeViterbi  += timeViterbi
          }
          outPw.println(ExportFormatParser.toString(tree, i))
        }

        if(cmd_args.tags_file == null){
          for( (sent, i) <- allSents.zipWithIndex ){
            processParseResult(i, model.parse(sent))
          }
        }else{
          val allTags = SuperTaggingModel.loadTokens(cmd_args.tags_file).toList
          for( ((sent, i), tags) <- allSents.zipWithIndex zip allTags ){
            assert(sent.size == tags.size)
            processParseResult(i, model.parse(sent, tags))
          }
        }

        statsPw   .println("totalTimedSents"  +"\t"+(    totalTimedSents                  ))
        System.err.println("totalTimedSents"  +"\t"+(    totalTimedSents                  ))

        statsPw   .println("totalTimeNeural"  +"\t"+(    totalTimeNeural                  ))
        statsPw   .println("totalTimeLabeling"+"\t"+(  totalTimeLabeling                  ))
        statsPw   .println("totalTimeParsing" +"\t"+(   totalTimeParsing                  ))
        statsPw   .println("totalTimeViterbi" +"\t"+(   totalTimeViterbi                  ))
        System.err.println("totalTimeNeural"  +"\t"+(    totalTimeNeural                  ))
        System.err.println("totalTimeLabeling"+"\t"+(  totalTimeLabeling                  ))
        System.err.println("totalTimeParsing" +"\t"+(   totalTimeParsing                  ))
        System.err.println("totalTimeViterbi" +"\t"+(   totalTimeViterbi                  ))

        statsPw   .println("avgTimeNeural"    +"\t"+(    totalTimeNeural /totalTimedSents ))
        statsPw   .println("avgTimeLabeling"  +"\t"+(  totalTimeLabeling /totalTimedSents ))
        statsPw   .println("avgTimeParsing"   +"\t"+(   totalTimeParsing /totalTimedSents ))
        statsPw   .println("avgTimeViterbi"   +"\t"+(   totalTimeViterbi /totalTimedSents ))
        System.err.println("avgTimeNeural"    +"\t"+(    totalTimeNeural /totalTimedSents ))
        System.err.println("avgTimeLabeling"  +"\t"+(  totalTimeLabeling /totalTimedSents ))
        System.err.println("avgTimeParsing"   +"\t"+(   totalTimeParsing /totalTimedSents ))
        System.err.println("avgTimeViterbi"   +"\t"+(   totalTimeViterbi /totalTimedSents ))

        System.err.println("parsing ended at "+Global.currentTimeHumanFormat)

        outPw.close()
        if(statsPw!=null)
          statsPw.close()
      case None =>
        System.err.println("You didn't specify all the required arguments")
        System.exit(-1)
    }
  }

}
