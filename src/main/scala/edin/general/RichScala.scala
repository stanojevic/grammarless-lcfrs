package edin.general

import scala.language.implicitConversions

object RichScala {
  implicit class MyRichAny[A](self:A){
    @inline def ?|>     (p: A => Boolean, f: A => A) : A         = if (p(self)) f(self) else self
    @inline def  |>  [B](f: A => B)                  : B         = f(self)
    @inline def =|>  [B](f: A => B)                  : B         = f(self)
    @inline def ??   [B](p: A => Boolean, f: A => B) : Option[B] = if(p(self)) Some(f(self)) else None
    @inline def some                                 : Option[A] = Some(self)
  }
  implicit class MyrichMap[A, B](map:Map[A, B]){
    def mapKeys[C](f: A => C): Map[C, B] = map.map { case (a, b) => (f(a), b) }
    def updateAtKey(key: A, f: B => B): Map[A, B] = map.get(key).fold(map)(value => map.updated(key, f(value)))
  }
  class IfTrue[A](b: => Boolean, t: => A) { def |(f: => A) = if (b) t else f }
  class MakeIfTrue(b: => Boolean) { def ?[A](t: => A) = new IfTrue[A](b,t) }
  implicit def autoMakeIfTrue(b: => Boolean) = new MakeIfTrue(b)
}
