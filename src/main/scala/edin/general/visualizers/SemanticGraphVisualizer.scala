package edin.general.visualizers

import java.io.{File, FileInputStream, FileOutputStream, PrintWriter}

import edin.general.visualizers.DepsVisualizer.DepsDesc

object SemanticGraphVisualizer {

  private def openViewer(file:String) : Unit = {
    val filename = new File(file).toPath
    val xdgCmd = System.getProperty("os.name") match {
      case "Linux" => s"nohup xdg-open $filename"
      case _       => s"open           $filename"
    }
    Runtime.getRuntime.exec(xdgCmd)
    val seconds = 5
    Thread.sleep(seconds*100)
  }

  def visualize(dd:DepsDesc, graphLabel:String="", fileType:String="pdf") : Unit = {
    val file = File.createTempFile(s"visual_${graphLabel.replace(" ", "_")}_", "."+fileType, null)
    file.deleteOnExit()
    saveVisual(dd, file, graphLabel=graphLabel, fileType=fileType)

    val filename = file.toPath
    val xdgCmd = System.getProperty("os.name") match {
      /** run this if you use evince and have big pdfs: gsettings set org.gnome.Evince page-cache-size 500 */
      case "Linux" => s"nohup xdg-open $filename"
      case _       => s"nohup open     $filename"
    }
    Runtime.getRuntime.exec(xdgCmd)
    val seconds = 2
    Thread.sleep(seconds*100)
  }

  def saveVisual(dd:DepsDesc, outFile:File, graphLabel:String="", fileType:String="pdf") : Unit = {
    outFile.getName.split(".").lastOption.foreach{ x =>
      if(x != fileType){
        sys.error(s"cannot save $fileType into file with extension $x")
      }
    }
    val dotString = toDotString(dd, graphLabel)

    val tmpDotFile = File.createTempFile("visual", ".dot", null)
    tmpDotFile.deleteOnExit()
    val tmpFileName = tmpDotFile.getPath
    val pw = new PrintWriter(tmpDotFile)
    pw.println(dotString)
    pw.close()

    val dotCmd = s"dot -T$fileType $tmpFileName -O"
    val pDot = Runtime.getRuntime.exec(dotCmd)
    pDot.waitFor()
    copyFile(s"$tmpFileName.$fileType", outFile.getAbsolutePath)
    new File(tmpFileName).delete()
    new File(s"$tmpFileName.$fileType").delete()
  }

  private def copyFile(src:String, dest:String) : Unit = {
    val inputChannel = new FileInputStream(src).getChannel
    val outputChannel = new FileOutputStream(dest).getChannel
    outputChannel.transferFrom(inputChannel, 0, inputChannel.size())
    inputChannel.close()
    outputChannel.close()
  }

  private def toDotString(dd:DepsDesc, graphLabel:String="") : String = {
    val top = dd.words.zipWithIndex.map{ case (word, i) =>
      "node"+i+"[label=\""+escapeForDot(word)+"\"];"
    }.mkString("\n")
    val middle = dd.deps.map{ case (hI, dI, label, _) =>
      "\tnode"+hI+" -> node"+dI+" [label=\""+label+"\"];"
    }.mkString("\n")
    s"digraph G {\n$top\n$middle\n}"
  }

  private def escapeForDot(s:String) : String =
    s.replace("\\", "\\\\")

}
