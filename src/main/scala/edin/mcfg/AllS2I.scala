package edin.mcfg

import java.io.{FileInputStream, FileOutputStream, ObjectInputStream, ObjectOutputStream}
import edin.nn.model.Any2Int

class AllS2I(
              val w2i : Any2Int[String], // word     2 int
              val t2i : Any2Int[String], // tag      2 int
              val n2i : Any2Int[String], // non-term 2 int
              val coreNTdesc : Array[Array[Int]]
            ) {

  import AllS2I.OUT_FN

  def save(modelDir:String) : Unit = {
    val fh = new ObjectOutputStream(new FileOutputStream(s"$modelDir/$OUT_FN"))
    fh.writeObject(w2i)
    fh.writeObject(t2i)
    fh.writeObject(n2i)
    fh.writeObject(coreNTdesc)
    fh.close()
  }
}

object AllS2I {

  val OUT_FN = "s2i.serialized"

  def load(modelDir:String) : AllS2I = {
    val fh = new ObjectInputStream(new FileInputStream(s"$modelDir/$OUT_FN"))
    val w2i = fh.readObject().asInstanceOf[Any2Int[String]]
    val t2i = fh.readObject().asInstanceOf[Any2Int[String]]
    val n2i = fh.readObject().asInstanceOf[Any2Int[String]]
    val coreNTdesc = fh.readObject().asInstanceOf[Array[Array[Int]]]
    fh.close()

    new AllS2I(
      w2i = w2i,
      t2i = t2i,
      n2i = n2i,
      coreNTdesc = coreNTdesc,
    )
  }

}

