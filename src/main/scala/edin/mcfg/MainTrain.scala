package edin.mcfg

import edin.general.Global
import edin.nn.DynetSetup
import edin.nn.model.TrainingController

object MainTrain {

  // CODE_DIR=/home/milos/Projects/CCG-translator
  // cd $CODE_DIR ; sbt assembly ; cd -
  // mkdir -p transformed
  // convert Tiger for training & split
  //   treetools transform originals_extracted/tiger_2.2/tiger_release_aug07.corrected.16012013.xml transformed/tiger_2.2.export  --src-format tigerxml --dest-format export --trans root_attach --split rest_5000#_5000#
  //   cd transformed
  //   mkdir tiger_2.2_preprocessed
  //   mv tiger_2.2.* tiger_2.2_preprocessed
  //   cd tiger_2.2_preprocessed
  //   mv *.export.0 train.export
  //   mv *.export.1 dev.export
  //   mv *.export.2 test.export
  //   $CODE_DIR/scripts/run.sh edin.constituency.MainTagUtils --command renumber_export_trees --in-trees dev.export --out-trees dev.export
  //   $CODE_DIR/scripts/run.sh edin.constituency.MainTagUtils --command renumber_export_trees --in-trees test.export --out-trees test.export
  //   for C in dev train test ; do echo $C ; $CODE_DIR/scripts/run.sh edin.constituency.MainTagUtils --command extract_tags_and_words --in-trees ${C}.export --out-tags ${C}.postags --out-words ${C}.words ; done
  // convert Negra for training & split
  //   iconv -f latin1 -t UTF8 originals_extracted/negra/negra-corpus.export > originals_extracted/negra/negra-corpus.utf8.export
  //   treetools transform originals_extracted/negra/negra-corpus.utf8.export transformed/negra.export  --src-format export --dest-format export --trans root_attach --split 80%_10%_10%
  //   cd transformed
  //   mkdir negra_preprocessed
  //   mv negra.* negra_preprocessed
  //   cd negra_preprocessed
  //   mv *.export.0 train.export
  //   mv *.export.1 dev.export
  //   mv *.export.2 test.export
  //   $CODE_DIR/scripts/run.sh edin.constituency.MainTagUtils --command renumber_export_trees --in-trees dev.export --out-trees dev.export
  //   $CODE_DIR/scripts/run.sh edin.constituency.MainTagUtils --command renumber_export_trees --in-trees test.export --out-trees test.export
  //   # attach punctuation back to the root if evaluating on non-transformed data with --trans punctuation_root
  //   for C in dev train test ; do echo $C ; $CODE_DIR/scripts/run.sh edin.constituency.MainTagUtils --command extract_tags_and_words --in-trees ${C}.export --out-tags ${C}.postags --out-words ${C}.words ; done
  // convert DPTB for training & split
  //   mkdir transformed/dptb_preprocessed
  //   cd transformed/dptb_preprocessed
  //   cp ../../originals_zips/discontinuous_penn_treebank_DPTB/dptb_split.tar.gz .
  //   tar xfvz dptb_split.tar.gz
  //   # convert from discbracket to export format
  //   for X in *.discbracket ; do echo $X ; Y=$(echo $X | sed "s/.discbracket/.export/") ; discodop treetransforms --inputfmt=discbracket --outputfmt=export $X $Y ; done
  //   $CODE_DIR/scripts/run.sh edin.constituency.MainTagUtils --command renumber_export_trees --in-trees dev.export  --out-trees dev.export
  //   $CODE_DIR/scripts/run.sh edin.constituency.MainTagUtils --command renumber_export_trees --in-trees test.export --out-trees test.export
  //   for C in dev train test ; do echo $C ; $CODE_DIR/scripts/run.sh edin.constituency.MainTagUtils --command extract_tags_and_words --in-trees ${C}.export --out-tags ${C}.postags --out-words ${C}.words ; done
  //   rm *.discbracket dptb_split.tar.gz
  // compress everything
  //   cd transformed
  //   zip -r disc_treebanks.zip dptb_preprocessed  negra_preprocessed  tiger_2.2_preprocessed
  //   scp disc_treebanks.zip mstanoje@pataphysique.inf.ed.ac.uk:/disk/scratch_big/mstanoje/Data
  // evaluate with discodop
  //   git clone --recursive git://github.com/andreasvc/disco-dop.git
  //   cd disco-dop
  //   pip3 install -r requirements.txt
  //   make install
  //   export PATH=$HOME/.local/bin:$PATH
  //   discodop eval gold.export pred.export --disconly

  // TODO check if ExportFormatParser.toString produces ROOT node

  // TODO is n^4 good enough even for long sentences?

  // TODO list:
  // TODO - save only best k checkpoints
  // TODO - checkpoint averaging
  // TODO - Matrix-Tree Theorem
  // TODO - MST by Tarjan
  // TODO - unsupervised masked tagger

  // TODO postags in incremental revealing just like in RNNG

  case class CMDargs(
                      model_dir                      : String    =  null,
                      train_file                     : String    =  null,
                      dev_file                       : String    =  null,
                      epochs                         : Int       =  2000,
                      hyper_params_file              : String    =  null,
                      devices                        : String    =  null,
                    )

  def main(args:Array[String]) : Unit = {

    val parser = new scopt.OptionParser[CMDargs]("MCFL-2 parser") {
      head("MCFL-2 parser", "0.1")
      opt[ String   ]( "model-dir"                 ).action((x,c) => c.copy( model_dir                      = x )).required()
      opt[ String   ]( "train-file"                ).action((x,c) => c.copy( train_file                     = x )).required()
      opt[ String   ]( "dev-file"                  ).action((x,c) => c.copy( dev_file                       = x )).required()
      opt[ Int      ]( "epochs"                    ).action((x,c) => c.copy( epochs                         = x ))
      opt[ String   ]( "hyper-params-file"         ).action((x,c) => c.copy( hyper_params_file              = x )).required()
      opt[ String   ]( "devices"                   ).action((x,c) => c.copy( devices                        = x ))
      help("help").text("prints this usage text")
    }

    parser.parse(args, CMDargs()) match {
      case Some(cmd_args) =>
        Global.printProcessId()

        DynetSetup.init_dynet(devices = cmd_args.devices)

        val model = new ParsingModel()
        new TrainingController(
          continueTraining = false,
          epochs           = cmd_args.epochs,
          trainFile        = cmd_args.train_file,
          devFile          = cmd_args.dev_file,
          modelDir         = cmd_args.model_dir,
          hyperFile        = cmd_args.hyper_params_file,
          modelContainer   = model
        ).train()
      case None =>
        System.err.println("You didn't specify all the required arguments")
        System.exit(-1)
    }
  }

}
