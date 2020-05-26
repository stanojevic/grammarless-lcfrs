package edin.mcfg

import edin.general.Global
import edin.nn.DynetSetup
import edin.nn.model.{ModelContainer, TrainingController}

object MainTrainPosttrain {

  // convert Tiger for training & split
  //   treetools transform originals_extracted/tiger_2.2/tiger_release_aug07.corrected.16012013.xml transformed/tiger_2.2.export  --src-format tigerxml --dest-format export --trans root_attach --split rest_5000#_5000#
  // convert Negra for training & split
  //   iconv -f latin1 -t UTF8 originals_extracted/negra/negra-corpus.export > originals_extracted/negra/negra-corpus.utf8.export
  //   treetools transform originals_extracted/negra/negra-corpus.export transformed/negra.export  --src-format export --dest-format export --trans root_attach --split 80%_10%_10%
  // attach punctuation back to the root if evaluating on non-transformed data with --trans punctuation_root
  // rename the resulting files
  //   for X in transformed/*.export.0 ; do N=$(echo $X | sed "s/\.export\.0\$/.train.export/") ; mv $X $N ; done
  //   for X in transformed/*.export.1 ; do N=$(echo $X | sed "s/\.export\.1\$/.dev.export/"  ) ; mv $X $N ; done
  //   for X in transformed/*.export.2 ; do N=$(echo $X | sed "s/\.export\.2\$/.test.export/" ) ; mv $X $N ; done
  // evaluate with discodop

  case class CMDargs(
                      old_model_dir                 : String    = null,
                      new_model_dir                  : String    = null,
                      train_file                     : String    = null,
                      dev_file                       : String    = null,
                      epochs                         : Int       =  100,
                      devices                        : String    = null,
                    )

  def main(args:Array[String]) : Unit = {

    val parser = new scopt.OptionParser[CMDargs]("MCFL-2 parser") {
      head("MCFL-2 parser", "0.1")
      opt[ String   ]( "old-model-dir"             ).action((x,c) => c.copy( old_model_dir                  = x )).required()
      opt[ String   ]( "new-model-dir"             ).action((x,c) => c.copy( new_model_dir                  = x )).required()
      opt[ String   ]( "train-file"                ).action((x,c) => c.copy( train_file                     = x )).required()
      opt[ String   ]( "dev-file"                  ).action((x,c) => c.copy( dev_file                       = x )).required()
      opt[ Int      ]( "epochs"                    ).action((x,c) => c.copy( epochs                         = x )).required()
      opt[ String   ]( "devices"                   ).action((x,c) => c.copy( devices                        = x ))
      help("help").text("prints this usage text")
    }

    parser.parse(args, CMDargs()) match {
      case Some(cmd_args) =>
        Global.printProcessId()
        DynetSetup.init_dynet(devices = cmd_args.devices)

        val model = new ParsingModel()
        model.loadFromModelDir(cmd_args.old_model_dir)
        new TrainingController(
          continueTraining = false,
          epochs           = cmd_args.epochs,
          trainFile        = cmd_args.train_file,
          devFile          = cmd_args.dev_file,
          modelDir         = cmd_args.new_model_dir,
          hyperFile        = s"${cmd_args.old_model_dir}/${ModelContainer.HYPER_PARAMS_FN}",
          modelContainer   = model
        ).train()
      case None =>
        System.err.println("You didn't specify all the required arguments")
        System.exit(-1)
    }
  }

}
