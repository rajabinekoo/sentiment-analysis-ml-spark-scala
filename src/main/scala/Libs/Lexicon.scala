package Libs

import scala.collection.mutable

import org.apache.spark.sql.functions.first
import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StructField, StructType}

object Lexicon {
  private val statusOfSentimentWords: mutable.Map[String, Double] = mutable.Map()

  def init(spark: SparkSession): mutable.Map[String, Double] = {
    val wordsMap: mutable.Map[String, Double] = mutable.Map()
    spark.read.textFile(Configs.VaderLexiconPath)
      .foreach(line => {
        val strRow = line.toString
        val parts = strRow.split("\t")
        if (parts.length >= 2) {
          val word = parts(0).trim
          val polarityStr = parts(1).trim
          try {
            val polarity = polarityStr.toDouble
            wordsMap.put(word, if (polarity > 0) 1.0 else if (polarity < 0) 0.0 else 2.0)
            print("")
          } catch {
            case _: NumberFormatException => print("")
          }
        }
      })
    wordsMap
  }

  def extractSentimentWordsCountFeature(spark: SparkSession, df: DataFrame): DataFrame = {
    val wordSentimentRDD = df.select("review_id", "text").rdd.flatMap { row =>
      val reviewId = row.getLong(0)
      val text = row.getString(1).toLowerCase().replaceAll("[^a-zA-Z\\s]", "").split("\\s+")
      text.filter(_.nonEmpty)
        .map(word => (reviewId, statusOfSentimentWords.getOrElse(word, -1.0)))
        .filter(_._2 != -1.0)
    }

    val sentimentCountsRDD = wordSentimentRDD
      .map { case (reviewId, sentiment) => ((reviewId, sentiment), 1) }
      .reduceByKey { case (count1, count2) => count1 + count2 }
      .map { case ((reviewId, sentiment), count) => Row(reviewId, sentiment, count) }

    val schema = StructType(Seq(
      StructField("review_id", LongType, nullable = false),
      StructField("sentiment_value", DoubleType, nullable = false),
      StructField("count", IntegerType, nullable = false)
    ))

    val sentimentCountsDF = spark.createDataFrame(sentimentCountsRDD, schema)
      .groupBy("review_id")
      .pivot("sentiment_value", Seq(1.0, 0.0, 2.0))
      .agg(first("count"))
      .withColumnRenamed("1.0", Configs.PositiveWordsCount)
      .withColumnRenamed("0.0", Configs.NegativeWordsCount)
      .withColumnRenamed("2.0", Configs.NeutralWordsCount)
      .na.fill(0)

    df.join(sentimentCountsDF, Seq("review_id"), "left").na.fill(0).drop("review_id")
  }
}
