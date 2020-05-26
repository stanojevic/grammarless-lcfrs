package edin.nn.layers

import edu.cmu.dynet.Expression

trait Layer {

  final def apply(x:Expression) : Expression = apply(x, Nil)
  def apply(x:Expression, targets:List[Int]) : Expression

}
