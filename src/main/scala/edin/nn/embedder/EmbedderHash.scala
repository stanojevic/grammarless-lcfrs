package edin.nn.embedder

import edin.nn.DyFunctions._
import edin.nn.DynetSetup
import edin.nn.layers.Dropout
import edin.nn.model.YamlConfig
import edu.cmu.dynet.{Expression, ParameterCollection}
import org.apache.commons.math3.primes.Primes

/**
  * implements Hash Embeddings from https://arxiv.org/pdf/1709.03933.pdf
  * with extensions that use universal hashing proposed by https://github.com/YannDubs/Hash-Embeddings
  * it achieves low probability of collision by using multiple random hashes
  * and learning not to value some input using importance weights
  */
sealed case class EmbedderHashConfig[K](
                                             outDim            : Int,
                                             k                 : Int, // distinct hash functions (and component vectors)
                                             B                 : Int, // size of pool of embedding vectors
                                             K                 : Int, // size of pool of importance vectors; set it to B is you want to learn importance of words&hashes; set it to 1 if you want only to learn importance of hashes
                                             dropout           : Float=0f
                                           ) extends EmbedderConfig[K] {
  def construct()(implicit model: ParameterCollection) = new EmbedderHash[K](this)
}

object EmbedderHashConfig{

  def fromYaml[K](conf:YamlConfig) : EmbedderConfig[K] =
    EmbedderHashConfig[K](
      outDim            = conf("out-dim").int,
      k                 = conf("k").int,
      B                 = conf("B").int,
      K                 = conf("K").int,
      dropout           = conf.getOrElse("dropout", 0f),
    )

}

object EmbedderHash{

  def main(args:Array[String]) : Unit = {
    DynetSetup.init_dynet()
    implicit val model = new ParameterCollection()

    val emb = EmbedderHashConfig[String](
      outDim = 13,
      k = 10,
      B = 2,
      K = 1
    ).construct()

    val vocSize = 5
    println("probability of total collinsions : "+emb.probabilityOfTotalCollision(    vocSize))
    println("expected #  of total collinsions : "+emb.expectedNumberOfTotalCollisions(vocSize))
    emb(List(
      "Ov0198401983kva recd4302914a",
      "Ovo je nekakva recd4302914a",
      "Ovo je nekakva reca",
      "Ovo je nekakva recenica",
      "Ovo je neakva recenica"
    )).printWithVals()
  }
}

class EmbedderHash[T](config:EmbedderHashConfig[T])(implicit model: ParameterCollection) extends Embedder[T] {

  private val E = model.addLookupParameters(config.B, config.outDim)
  private val P = model.addLookupParameters(config.K, config.k)

  private val hashes  : List[Int => Int] = constructKHashesCarterWegman(seed=7, k=config.k, m=config.B)
  private val impHash :      Int => Int  = constructKHashesCarterWegman(seed=3, k=1       , m=config.K).head

  private val drop = Dropout(config.dropout)

  override val outDim: Int = config.outDim

//  override def apply(x: T): Expression = embedHash(x.hashCode())

  override def apply(x: List[T]): Expression = embedHash(x.map(_.hashCode()))

  def embedHash(hashVal:Int) : Expression = embedHash(List(hashVal))

  def embedHash(hashVals:List[Int]) : Expression = {
    val imp = P(hashVals map impHash)
    val components = for(hash <- hashes) yield E(hashVals map hash)
    drop(concatCols(components:_*)*imp)
  }

  def probabilityOfTotalCollision(inputVocSize:Int) : Double = // this is only approximation for large K, B etc.
    1d - math.exp(-inputVocSize.toDouble/(config.K*math.pow(config.B, config.k)))

  def expectedNumberOfTotalCollisions(inputVocSize:Int) : Double =
    inputVocSize*probabilityOfTotalCollision(inputVocSize)

  /**
    * Implements Carter & Wegman 1979 algorithm https://en.wikipedia.org/wiki/Universal_hashing#Hashing_integers
    * for constructing universal hashing functions for integers
    * If you want a faster version you should implement multiply-add-shift algoritham by Dietzfelbinger et al. 1997
    * that manipulates bits and doesn't require prime numbers
    * @param k is a number of hash functions to return
    * @param m is a number of bins (output)            */
  def constructKHashesCarterWegman(seed:Int, k:Int, m:Int) : List[Int => Int] = {
    val rng = new scala.util.Random(seed)
    val p = if(m<=0)
      sys.error(s"output vocabulary cannot be negative or zero number: $m")
    else if(m<17) // condition m==1 is necessary for corner cases when we don't need hashing (output will always be 0)
      17          // condition m==2 so that it doesn't crash on rng.nextInt(2-2) // condition <17 for more randomness for small numbers
    else
      Primes.nextPrime(m)
    val as = for(_ <- 0 until k) yield rng.nextInt(p-2)+1
    val bs = for(_ <- 0 until k) yield rng.nextInt(p-1)
    for((a, b) <- (as zip bs).toList)
      yield {x:Int => (math.abs(a*x+b)%p)%m} // math.abs is necessary because of the potential overflow
  }

}

