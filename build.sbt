name := "CCG-translator"
version := "0.2"
organization := "milos.unlimited"

scalaVersion := "2.12.10"

unmanagedBase := baseDirectory.value / "lib"

resolvers += "jitpack" at "https://jitpack.io"

libraryDependencies += "org.scala-lang.modules" %% "scala-parser-combinators" % "1.1.2"
libraryDependencies += "org.scalatest" %% "scalatest" % "3.2.0-M1" % "test"

//libraryDependencies += "com.github.wookietreiber" %% "scala-chart" % "latest.integration" // for plotting
//libraryDependencies += "com.itextpdf" % "itextpdf" % "5.5.6" // so plotting can be done in pdf

libraryDependencies += "com.github.scopt" %% "scopt" % "3.7.1" // for cmd line argument parsing
libraryDependencies += "org.yaml" % "snakeyaml" % "1.8"        // for yaml
libraryDependencies += "io.spray" %%  "spray-json" % "1.3.5"   // for json

libraryDependencies += "edu.stanford.nlp" % "stanford-corenlp" % "3.9.2"    // for Penn tokenizer
dependencyOverrides += "javax.xml.bind" % "jaxb-api" % "2.4.0-b180830.0359" // resolving stanford caused version conflict
dependencyOverrides += "xml-apis" % "xml-apis" % "2.0.2"                    // resolving stanford caused version conflict
dependencyOverrides += "joda-time" % "joda-time" % "2.9.4"                  // resolving stanford caused version conflict

libraryDependencies += "org.typelevel" %% "spire" % "0.17.0-M1"   // for faster cfor and cforRange loops

libraryDependencies += "com.lihaoyi" %% "ammonite-ops" % "1.7.4"

libraryDependencies += "org.mapdb" % "mapdb" % "3.0.7"  // for peristant maps that i use for storing pretrained embeddings

libraryDependencies += "org.apache.commons" % "commons-math3" % "3.6.1" // for computing prime numbers in hashing function of hash emb & SVD in ortho_init

// libraryDependencies += "org.jheaps" % "jheaps" % "0.10" // for Fibonacci heaps
libraryDependencies += "com.github.d-michail" % "jheaps" % "fd5c4c15ee" // jheaps version with support for heapify handlers -- thanks to jitpack

libraryDependencies += "com.github.adamheinrich" % "native-utils" % "e6a394896"	// for packing dynamic libraries (.so, .dll) in jar -- thanks to jitpack

mainClass             in Compile  := None         // so assembly doesn't complain of having multiple main classes
test                  in assembly := {}           // so assembly doesn't trigger unnecessary testing
logLevel              in assembly := Level.Error  // so assembly doesn't complain of multiple versions of same jars
assemblyMergeStrategy in assembly := {
  case PathList("META-INF", xs @ _*) => MergeStrategy.discard
  case x => MergeStrategy.first
}

scalacOptions ++= Seq("-deprecation", "-feature") // for more informative output of a compiler

// OPTIMIZATION
val optimized = List("OPTIMIZED", "OPTIMIZE").exists(x => sys.env.contains(x) && sys.env(x) == "true")
if(optimized){
  System.err.println("\n--- COMPILING OPTIMIZED VERSION ---\n")
  scalacOptions ++= Seq("-opt:l:inline", "-opt-inline-from:**", "-opt:l:method")
  scalacOptions  += "-opt-warnings:none"
}else{
  scalacOptions  += "-opt-warnings:none"
}

