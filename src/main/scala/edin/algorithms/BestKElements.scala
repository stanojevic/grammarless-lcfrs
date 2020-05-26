package edin.algorithms

import scala.reflect.ClassTag

import spire.algebra.Order.{fromOrdering, reverse}
import spire.math.Selection.quickSelect
import spire.math.Sorting.quickSort

object BestKElements {

  implicit class IterableBestKBy[T](xs:Iterable[T])(implicit tag:ClassTag[T]){

//    import scala.collection.mutable.{PriorityQueue => MutPriorityQueue}
//    @inline
//    def bestKOrderingHeap(k:Int)(ordering:Ordering[T]) : List[T] = {
//      val q = new MutPriorityQueue[T]()( -ordering.compare(_, _) )
//      for(x <- xs){
//        q.enqueue(x)
//        if(q.size > k){
//          q.dequeue()
//        }
//      }
//      var res = List[T]()
//      while(q.nonEmpty){
//        res ::= q.dequeue()
//      }
//      res
//    }

    ///  spire implementation
    @inline
    def bestKOrdering(k:Int)(ordering:Ordering[T]) : List[T] = {
      assert(k>=0)
      if(k==1){
        implicit val ord = ordering
        List(xs.max)
      }else{
        implicit val order = reverse(fromOrdering(ordering))
        val a = xs.toArray
        quickSelect(a, k)
        val b = a.take(k)
        quickSort(b)
        b.toList
      }
    }

    @inline
    def worstKOrdering(k:Int)(ordering:Ordering[T]) : List[T] = bestKOrdering(k)(ordering.reverse)

    @inline
    def bestKBy[X](k:Int)(key: T => X)(implicit cv: X => Ordered[X]) : List[T] = bestKOrdering(k)(buildComparator(key))

    @inline
    def worstKBy[X](k:Int)(key: T => X)(implicit cv: X => Ordered[X]) : List[T] = worstKOrdering(k)(buildComparator(key))

    @inline
    private def buildComparator[X](key: T => X)(implicit cv: X => Ordered[X]) : Ordering[T] = key(_) compare key(_)

  }

  implicit class IterableBestK[T](xs:Iterable[T])(implicit cv: T => Ordered[T], tag:ClassTag[T]){
    @inline
    def bestK(k:Int) : List[T] = xs.bestKBy(k)(identity)

    @inline
    def worstK(k:Int) : List[T] = xs.worstKBy(k)(identity)
  }

}

