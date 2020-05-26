package edin.nn.layers

import edu.cmu.dynet.Expression
import edin.nn.DyFunctions._

final class IdentityLayer private () extends Layer {

  override def apply(x: Expression, targets: List[Int]): Expression = targets match {
    case Nil => x
    case _ => selectRows(x, targets)
  }

}

object IdentityLayer{

  def apply() : IdentityLayer = new IdentityLayer()

}

