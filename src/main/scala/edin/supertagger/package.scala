package edin

package object supertagger {

  val SUPERTAGGER_NAME = "SUPER_WHATEVER"
  val SUPERTAGGER_VERSION = 0.2

  type SentEmbedding = List[Array[Float]]

  sealed case class TrainInst(in_words:List[String], out_tags:List[String], in_tags:List[String])

}
