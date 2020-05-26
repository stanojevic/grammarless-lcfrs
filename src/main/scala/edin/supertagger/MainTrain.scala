package edin.supertagger

import edin.general.Global
import edin.nn.DynetSetup
import edin.nn.contextualized.External
import edin.nn.model.TrainingController

object MainTrain {


  case class CMDargs(
                      model_dir                      : String    = null,
                      train_file                     : String    = null,
                      dev_file                       : String    = null,
                      epochs                         : Int       =  100,
                      hyper_params_file              : String    = null,
                      out_tag_extension              : String    = "postags",
                      external_contextual_embeddings : String    = null,
                      dynet_mem                      : String    = null,
                      dynet_devices                  : String    = null,
                    )

  def main(args:Array[String]) : Unit = {
    val parser = new scopt.OptionParser[CMDargs](SUPERTAGGER_NAME) {
      head(SUPERTAGGER_NAME, SUPERTAGGER_VERSION.toString)
      opt[ String   ]( "model-dir"                          ).action((x,c) => c.copy( model_dir                      = x )).required()
      opt[ String   ]( "train-file"                         ).action((x,c) => c.copy( train_file                     = x )).required()
      opt[ String   ]( "dev-file"                           ).action((x,c) => c.copy( dev_file                       = x )).required()
      opt[ Int      ]( "epochs"                             ).action((x,c) => c.copy( epochs                         = x )).required()
      opt[ String   ]( "out-tag-extension"                  ).action((x,c) => c.copy( out_tag_extension              = x ))
      opt[ String   ]( "hyper-params-file"                  ).action((x,c) => c.copy( hyper_params_file              = x )).required()
      opt[ String   ]( "external-embedding-contextual-file" ).action((x,c) => c.copy( external_contextual_embeddings = x ))
      opt[ String   ]( "dynet-devices"                      ).action((x,c) => c.copy( dynet_devices                  = x ))
      opt[ String   ]( "dynet-mem"                          ).action((x,c) => c.copy( dynet_mem                      = x ))
      help("help").text("prints this usage text")
    }

    parser.parse(args, CMDargs()) match {
      case Some(cmd_args) =>
        Global.printProcessId()

        DynetSetup.init_dynet(devices=cmd_args.dynet_devices, dynet_mem = cmd_args.dynet_mem)

        if(cmd_args.external_contextual_embeddings != null)
          External.loadEmbeddings(cmd_args.external_contextual_embeddings)

        val modelContainer = new SuperTaggingModel(outTagExtension = cmd_args.out_tag_extension)

        new TrainingController(
          continueTraining = false,
          epochs           = cmd_args.epochs,
          trainFile        = cmd_args.train_file,
          devFile          = cmd_args.dev_file,
          modelDir         = cmd_args.model_dir,
          hyperFile        = cmd_args.hyper_params_file,
          modelContainer   = modelContainer,
        ).train()
      case None =>
        System.err.println("You didn't specify all the required arguments")
        System.exit(-1)
    }
  }

}
