package edin.general

import java.util.regex.Pattern

object PTBtokenizer {

  import scala.collection.JavaConverters._
  def tokenize(sent:String) : List[String] =
    edu.stanford.nlp.process.PTBTokenizer.newPTBTokenizer(new java.io.StringReader(sent)).
      tokenize().asScala.toList.map(_.word)
//      tokenize().toArray().toList.asInstanceOf[List[edu.stanford.nlp.ling.Word]].map(_.word)

  def escapeBrackets(words:List[String]) : List[String] = words map {
    case "{" => "-LCB-"
    case "}" => "-RCB-"
    case "[" => "-LSB-"
    case "]" => "-RSB-"
    case "(" => "-LRB-"
    case ")" => "-RRB-"
    case x   => x
  }

  def main(args:Array[String]) : Unit =
    println(tokenize("This isn't a {sentence} by 'John and' John' 'and Milos' and Milos', 'with some' \"quotes\" and? [brackets] (made) with author's ...permission etc. etc. .").mkString(" "))

  // TODO figure out single quotes

  private val replacements: Seq[(Pattern, String)] = List(
    ("\\.\\.\\."            , " ... "  ), // three dots
    ("[,;:@#$%&?!]"         , " $0 "   ), // all kinds of punctuation
    ("""[\]\[(){}<>]"""     , " $0 "   ), // separating all possible brackets
    ("--"                   , " $0 "   ), // double dash
    ("\""                   , " '' "   ), // making quotes more uniform (opening and closing will be handled later)
    ("``"                   , " '' "   ), // making quotes more uniform (opening and closing will be handled later)
    ("'([sSmMdD]) "         , " '$1 "  ), // for possesives, I'm and he'd
    ("'(ll|re|ve|LL|RE|VE) ", " '$1 "  ), // for we're you'll you've etc.
    ("(n't|N'T) "           , " $1 "   ), // negation
    ("[?!.] *$"             , " $0 "   ), // punctuation in the end of the sentence
    (" \\{ "                , " -LCB- "),
    (" \\} "                , " -RCB- "),
    (" \\[ "                , " -LSB- "),
    (" \\] "                , " -RSB- "),
    (" \\( "                , " -LRB- "),
    (" \\) "                , " -RRB- "),
  ).map{
    case (a, b) => (Pattern.compile(a), b)
  }

  def tokenize2(sent: String) : List[String] =
    replacements.foldLeft(sent) {
      case (s, (p, r)) => p.matcher(s).replaceAll(r)
    }.trim.split("\\s+").toList.foldLeft((List[String](), false)){
      case ((tokens, false  ), "''") => (   "``"::tokens, true   )
      case ((tokens, true   ), "''") => (   "''"::tokens, false  )
      case ((tokens, isOpen ),    w) => (      w::tokens, isOpen )
    }._1.reverse

}
