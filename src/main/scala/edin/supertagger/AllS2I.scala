package edin.supertagger

import java.io._
import edin.nn.model.Any2Int

class AllS2I (
               val in_w2i  : Any2Int[String], // word              2 int
               val out_t2i : Any2Int[String], // supertag          2 int
               val in_t2i  : Any2Int[String], // auxiliary tags    2 int
             ){

  import AllS2I.OUT_FN

  def save(modelDir:String) : Unit = {
    val fh = new ObjectOutputStream(new FileOutputStream(s"$modelDir/$OUT_FN"))
    fh.writeObject(in_w2i)
    fh.writeObject(in_t2i)
    fh.writeObject(out_t2i)
    fh.close()
  }
}

object AllS2I {

  val OUT_FN = "s2i.serialized"

  def load(modelDir:String) : AllS2I = {
    val fh = new ObjectInputStream(new FileInputStream(s"$modelDir/$OUT_FN"))
    val in_w2i  = fh.readObject().asInstanceOf[Any2Int[String]]
    val in_t2i  = fh.readObject().asInstanceOf[Any2Int[String]]
    val out_t2i = fh.readObject().asInstanceOf[Any2Int[String]]
    fh.close()

    new AllS2I(
      in_w2i  = in_w2i,
      out_t2i = out_t2i,
      in_t2i  = in_t2i
    )
  }

}

