package edin.supertagger

import java.io.{File, PrintWriter}
import edin.general.Global
import edin.nn.DynetSetup
import edin.nn.model.{TrainingController, YamlConfig}
import scala.util.Random

object MainJackkniffing {

  private var tagsExtension = ".postags"

  case class CMDargs(
                      model_dir                      : String    =      null,
                      train_file                     : String    =      null,
                      dev_file                       : String    =      null,
                      test_file                      : String    =      null,
                      hyper_params_file              : String    =      null,
                      embeddings_contextual_external : String    =      null,
                      devices                        : String    =      null,
                      tagsExtension                  : String    = "postags",
                      epochs                         : Int       =   3908230, // random
                      folds                          : Int       =        10,
                    )

  def main(args:Array[String]) : Unit = {
    val parser = new scopt.OptionParser[CMDargs](SUPERTAGGER_NAME) {
      head(SUPERTAGGER_NAME, SUPERTAGGER_VERSION.toString)
      opt[ String   ]( "model-dir"         ).action((x,c) => c.copy( model_dir         = x )).required()
      opt[ String   ]( "train-file"        ).action((x,c) => c.copy( train_file        = x )).required()
      opt[ String   ]( "dev-file"          ).action((x,c) => c.copy( dev_file          = x )).required()
      opt[ String   ]( "test-file"         ).action((x,c) => c.copy( test_file         = x ))
      opt[ String   ]( "hyper-params-file" ).action((x,c) => c.copy( hyper_params_file = x )).required()
      opt[ String   ]( "devices"           ).action((x,c) => c.copy( devices           = x ))
      opt[ Int      ]( "epochs"            ).action((x,c) => c.copy( epochs            = x ))
      opt[ Int      ]( "folds"             ).action((x,c) => c.copy( folds             = x )).required()
      help("help").text("prints this usage text")
    }

    val cmd_args = parser.parse(args, CMDargs()).getOrElse(sys.error("You didn't specify all the required arguments"))
    Global.printProcessId()
    DynetSetup.init_dynet(devices=cmd_args.devices)

    this.tagsExtension = cmd_args.tagsExtension

    val useEarlyStopping = YamlConfig.fromFile(cmd_args.hyper_params_file)("trainer").getOrElse("early-stopping", false)
    require(useEarlyStopping || cmd_args.epochs!=CMDargs().epochs, "jackkniffing must be done either with early stopping or by specifying number of epochs")

    new File(cmd_args.model_dir).mkdirs()

    val tmpTrainFile  = cmd_args.model_dir+"/tmp_train_file"
    val predTrainFile = cmd_args.model_dir+"/jack_train_file"

    cp(s"${cmd_args.train_file}.words"         , s"$tmpTrainFile.words"          )
    cp(s"${cmd_args.train_file}.$tagsExtension", s"$tmpTrainFile.$tagsExtension" )

    System.err.println("JACKKNIFFING TRAINING THE MAIN MODEL AND PRODUCEING NEW DEV SET")
    mainModelPhase(
      trainFile       = tmpTrainFile,
      devFile         = cmd_args.dev_file,
      testFile        = cmd_args.test_file,
      modelDir        = cmd_args.model_dir,
      hyperParamsFile = cmd_args.hyper_params_file,
      epochs          = cmd_args.epochs
    )

    val taggedSents = scala.collection.mutable.Map[List[String], List[String]]()

    val allTrainWords = SuperTaggingModel.loadTokens(s"${cmd_args.train_file}.words")
    val allTrainTags  = SuperTaggingModel.loadTokens(s"${cmd_args.train_file}.$tagsExtension" )
    for(((foldTrain, foldTest), foldId) <- createFolds(cmd_args.folds, (allTrainWords zip allTrainTags).toList).zipWithIndex){
      System.err.println(s"JACKKNIFFING START PROCESSING FOLD ${foldId+1}/${cmd_args.folds}")
      val pwWords = new PrintWriter(s"$tmpTrainFile.words")
      val pwTags  = new PrintWriter(s"$tmpTrainFile.$tagsExtension")
      for((words, tags) <- foldTrain){
        pwWords.println(words mkString " ")
        pwTags.println( tags  mkString " ")
      }
      pwWords.close()
      pwTags.close()
      val model = trainModel(
        trainFile       = tmpTrainFile,
        devFile         = cmd_args.dev_file,
        modelDir        = cmd_args.model_dir+"/tmp_model",
        hyperParamsFile = cmd_args.hyper_params_file,
        epochs          = cmd_args.epochs
      )
      for((words, _) <- foldTest){
        taggedSents(words) = model.tagSingleSentArgmax(words)
      }
    }
    System.err.println(s"JACKKNIFFING STORING NEW JACKKNIFFED TRAIN SET")

    // val pwWords = new PrintWriter(s"$predTrainFile.words")
    val pwTags  = new PrintWriter(s"$predTrainFile.$tagsExtension")
    for(words <- SuperTaggingModel.loadTokens(cmd_args.train_file+".words")){
      // pwWords.println(words mkString " ")
      pwTags.println(taggedSents(words) mkString " ")
    }
    // pwWords.close()
    pwTags.close()
    new File(s"$tmpTrainFile.$tagsExtension").delete()
    new File(s"$tmpTrainFile.words"         ).delete()
  }

  private def mainModelPhase(trainFile:String, devFile:String, testFile:String, modelDir:String, hyperParamsFile:String, epochs:Int) : Unit = {
    var model = trainModel(trainFile, devFile, modelDir, hyperParamsFile, epochs)
    val predDevFile   = s"$modelDir/jack_dev_file"
    val predTestFile  = s"$modelDir/jack_test_file"

    {
      val allTents = SuperTaggingModel.loadTokens(s"$devFile.words").toList
      val allTags = model.tagManySentArgmax(allTents)
      // cp(s"$devFile.words", s"$predDevFile.words")
      val pw = new PrintWriter(s"$predDevFile.$tagsExtension")
      for(tags <- allTags){
        pw.println(tags mkString " ")
      }
      pw.close()
    }
    if(testFile != null){
      val allTents = SuperTaggingModel.loadTokens(s"$testFile.words").toList
      val allTags = model.tagManySentArgmax(allTents)
      // cp(s"$testFile.words", s"$predTestFile.words")
      val pw = new PrintWriter(s"$predTestFile.$tagsExtension")
      for(tags <- allTags){
        pw.println(tags mkString " ")
      }
      pw.close()
    }
    model=null
  }

  private def trainModel(trainFile:String, devFile:String, modelDir:String, hyperParamsFile:String, epochs:Int) : SuperTaggingModel = {
    val modelContainer = new SuperTaggingModel(inTagExtension = "", outTagExtension = tagsExtension)
    new TrainingController(
      continueTraining = false,
      epochs           = epochs,
      trainFile        = trainFile,
      devFile          = devFile,
      modelDir         = modelDir,
      hyperFile        = hyperParamsFile,
      modelContainer   = modelContainer,
    ).train()
    modelContainer
  }

  private def createFolds[I](folds:Int, trainData:List[I]) : List[(List[I], List[I])] = {
    val data = new Random(42).shuffle(trainData)
    val stepContinuous = data.size.toDouble/folds
    val initPoints = (0 until folds).map(_*stepContinuous).map(Math.ceil).map(_.toInt).toList
    val endPoints  = initPoints.tail :+ trainData.size

    val res = for{
      (start, end) <- initPoints zip endPoints
    } yield {
      val prefix = trainData.take(start)
      val middle = trainData.slice(start, end)
      val suffix = trainData.drop(end)
      (prefix++suffix, middle)
    }
    assert(res.map(_._2.size).sum == trainData.size)
    res
  }

  private def cp(source:String, target:String) : Unit = {
    import ammonite.ops.{cp => acp, rm, Path}
    val srcPath = Path(new File(source).getAbsolutePath)
    val tgtPath = Path(new File(target).getAbsolutePath)
    rm(tgtPath)
    acp(srcPath, tgtPath)
  }

}
