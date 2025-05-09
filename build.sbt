ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.12.20"

lazy val root = (project in file("."))
  .settings(
    name := "sentiment-analysis-ml-model"
  )

val sparkVersion = "3.5.5"

libraryDependencies ++= Seq(
  "org.apache.spark" %% "spark-core" % sparkVersion,
  "org.apache.spark" %% "spark-sql" % sparkVersion,
  "org.apache.spark" %% "spark-mllib" % sparkVersion,
  "com.datastax.spark" %% "spark-cassandra-connector" % "3.5.1",
  "com.johnsnowlabs.nlp" %% "spark-nlp" % "5.4.1",
  "com.amazonaws" % "aws-java-sdk-s3" % "1.12.773" exclude("com.fasterxml.jackson.core", "jackson-databind"),
  "com.fasterxml.jackson.module" %% "jackson-module-scala" % "2.15.2"
)