package edin.nn.layers

import edu.cmu.dynet.Expression
import edin.nn.DyFunctions._
import edin.nn.DynetSetup

/**
  * useful for maintaning the same mask across all usages in the minibatch
  * (all instances and all applications within an instance)
  *
  * usage of the same mask across a single instance is motivated by
  * "A Theoretically Grounded Application of Dropout inRecurrent Neural Networks"
  * however usage of the same mask in the whole minibatch could be a bit strange and some say cause variance
  * it can be modified via sameForWholeBatch argument
  */
object Dropout {

  def apply(
             dropProb:Float,
             sameForAllApplications:Boolean=false,
             sameForWholeBatch:Boolean=false
           ) : Dropout =
    new Dropout(
      dropProb               = dropProb,
      sameForAllApplications = sameForAllApplications,
      sameForWholeBatch      = sameForWholeBatch
    )

}

class Dropout private (dropProb:Float, sameForAllApplications:Boolean, sameForWholeBatch:Boolean) extends Layer {

  private var dropMask : Expression = _
  private var latestCG : Int = -1

  // ones of exp's shape
  private def ones(exp:Expression) : Expression = if(sameForWholeBatch) exp.pickBatchElem(0).ones else exp.ones

  override def apply(exp:Expression, targets:List[Int]) : Expression =
    if(dropoutIsEnabled && dropProb>0){
      if(sameForAllApplications){
        if(latestCG != DynetSetup.cg_id){
          latestCG = DynetSetup.cg_id
          dropMask = Expression.dropout(ones(exp), dropProb)
        }
        exp ⊙ dropMask
      }else{
        dropout(exp, dropProb)
      }
    }else{
      exp
    }

}

