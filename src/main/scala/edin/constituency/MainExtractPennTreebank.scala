package edin.constituency

import java.io.{File, PrintWriter}

import edin.algorithms.AutomaticResourceClosing.linesFromFile
import edin.constituency.representation.PennFormatParser

object MainExtractPennTreebank {

  case class CMDargs(
                      penn_mrg_dir : String    = null,
                      out_dir      : String    = null
                    )

  def main(args:Array[String]) : Unit = {

    val parser = new scopt.OptionParser[CMDargs](PROGRAM_NAME) {
      head(PROGRAM_NAME, PROGRAM_VERSION.toString)
      opt[String]("penn_mrg_dir").action((x, c) => c.copy(penn_mrg_dir = x)).required()
      opt[String]("out_dir"     ).action((x, c) => c.copy(out_dir      = x)).required()
    }

    parser.parse(args, CMDargs()) match {
      case Some(cmd_args) =>

        val out_dir = new File(cmd_args.out_dir)
        if (!out_dir.exists())
          out_dir.mkdir()

        val parseFiles = allParseFilesSorted(cmd_args.penn_mrg_dir)
        val trainFiles = parseFiles.filter { f =>
          val section = f.getParentFile.getName.toInt
          2 <= section && section <= 21
        }
        val devFiles = parseFiles.filter { f =>
          val section = f.getParentFile.getName.toInt
          section == 0
        }
        val testFiles = parseFiles.filter { f =>
          val section = f.getParentFile.getName.toInt
          section == 23
        }
        val otherFiles = parseFiles.filter { f =>
          val section = f.getParentFile.getName.toInt
          Set(1, 22, 24) contains section
        }
        val toyFiles = List(trainFiles.head)

        for ((typ, files) <- List(
          ("train", trainFiles),
          ("dev"  , devFiles  ),
          ("test" , testFiles ),
          ("other", otherFiles),
          ("toy"  , toyFiles  ))
        ) {
          val pw_trees = new PrintWriter(cmd_args.out_dir+"/"+typ+".mrg")
          val pw_pos = new PrintWriter(cmd_args.out_dir+"/"+typ+".postags")
          val pw_words = new PrintWriter(cmd_args.out_dir+"/"+typ+".words")
          files.flatMap(linesFromFile).foreach(pw_trees.println)
          for(tree <- files.toStream.map(_.getAbsolutePath).flatMap(PennFormatParser.fromFile)){
            pw_pos.println(tree.deleteEmptyNodes.leafsSorted.map(_.label).mkString(" "))
            pw_words.println(tree.deleteEmptyNodes.leafsSorted.map(_.word).mkString(" "))
          }
          pw_trees.close()
          pw_pos.close()
          pw_words.close()
        }
      case None =>
        System.err.println("You didn't specify all the required arguments")
        System.exit(-1)
    }
  }

  private def allParseFilesSorted(f:String) : List[File] =
    allParseFilesUnordered(new File(f)).sortBy(_.getName)

  private def allParseFilesUnordered(f:File) : List[File] =
    if(f.isDirectory)
      f.listFiles().toList.flatMap(allParseFilesUnordered)
    else if(f.getName endsWith ".mrg")
      List(f)
    else
      Nil

}
