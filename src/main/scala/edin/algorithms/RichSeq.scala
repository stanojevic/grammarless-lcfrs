package edin.algorithms

object RichSeq {

  implicit class Splitting[A](stream:Seq[A]){

    def splitByContentWithDrop   (p:A=>Boolean) : Stream[List[A]] = splitByContent(stream, separatorStrategy = Dropping    )(p)
    def splitByContentAsEnding   (p:A=>Boolean) : Stream[List[A]] = splitByContent(stream, separatorStrategy = AsEnding    )(p)
    def splitByContentAsBeginning(p:A=>Boolean) : Stream[List[A]] = splitByContent(stream, separatorStrategy = AsBeginning )(p)

    private sealed trait  SeparatorStrategy
    private case   object Dropping    extends SeparatorStrategy
    private case   object AsEnding    extends SeparatorStrategy
    private case   object AsBeginning extends SeparatorStrategy

    private def splitByContent(s:Seq[A], separatorStrategy: SeparatorStrategy)(p:A=>Boolean) : Stream[List[A]] =
      separatorStrategy match {
        case _ if s.isEmpty =>
          Stream.empty
        case Dropping =>
          val (prefix, rest) = s.span(x => ! p(x))
          lazy val subResult = if(rest.isEmpty) Stream.empty else splitByContent(rest.tail, separatorStrategy)(p)
          if(prefix.isEmpty) subResult
          else               Stream.cons(prefix.toList             , subResult)
        case AsEnding =>
          val (prefix, rest) = s.span(x => ! p(x))
          lazy val subResult = if(rest.isEmpty) Stream.empty else splitByContent(rest.tail, separatorStrategy)(p)
          if(prefix.isEmpty)    Stream.cons(List(rest.head)           , subResult)
          else if(rest.isEmpty) Stream.cons(prefix.toList             , subResult)
          else                  Stream.cons(prefix.toList :+ rest.head, subResult)
        case AsBeginning =>
          if(p(s.head)){
            val (prefix, rest) = s.tail.span(x => ! p(x))
            lazy val subResult = if(rest.isEmpty) Stream.empty else splitByContent(rest, separatorStrategy)(p)
            Stream.cons(s.head :: prefix.toList, subResult)
          }else{
            val (prefix, rest) = s.span(x => ! p(x))
            lazy val subResult = if(rest.isEmpty) Stream.empty else splitByContent(rest, separatorStrategy)(p)
            Stream.cons(          prefix.toList, subResult)
          }
      }

  }


}
