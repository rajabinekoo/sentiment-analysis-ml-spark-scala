package Libs

import scala.collection.mutable

import org.apache.spark.sql.{DataFrame, Row}
import org.apache.spark.sql.functions.{col, first}
import org.apache.spark.sql.types.{DoubleType, IntegerType, LongType, StructField, StructType}

trait Lexicon {
  val vader: mutable.Map[String, Double] = {
    val words: mutable.Map[String, Double] = mutable.Map()
    val lexiconRows = SparkInstance.singleton.read
      .option("delimiter", "\t")
      .csv(Configs.VaderLexiconPath)
      .toDF("word", "polarity", "std_dev", "scores_raw")
      .filter(col("word").isNotNull).collectAsList()
    lexiconRows.forEach(row => {
      try {
        val polarity = row.getString(1).toDouble
        val value = if (polarity > 0.1) 1.0 else if (polarity < -0.1) 0.0 else 2.0
        words.put(row.getString(0).trim().toLowerCase(), value)
      }
    })
    words
  }
}

object Lexicon extends Lexicon {
  def extractSentimentWordsCountFeature(df: DataFrame): DataFrame = {
    val wordSentimentRDD = df
      .select("id", "body").rdd.flatMap { row =>
        val reviewId = row.getLong(0)
        val text = row.getString(1).toLowerCase().replaceAll("[^a-zA-Z\\s]", "").split("\\s+")
        text.filter(_.nonEmpty)
          .map(word => (reviewId, vader.getOrElse(word, -0.1)))
          .filter(_._2 != -1.0)
      }

    val sentimentCountsRDD = wordSentimentRDD
      .map { case (reviewId, sentiment) => ((reviewId, sentiment), 1) }
      .reduceByKey { case (count1, count2) => count1 + count2 }
      .map { case ((reviewId, sentiment), count) => Row(reviewId, sentiment, count) }

    val schema = StructType(Seq(
      StructField("id", LongType, nullable = false),
      StructField("sentiment_value", DoubleType, nullable = false),
      StructField("count", IntegerType, nullable = false)
    ))

    val sentimentCountsDF = SparkInstance.singleton.createDataFrame(sentimentCountsRDD, schema)
      .groupBy("id")
      .pivot("sentiment_value", Seq(1.0, 0.0, 2.0))
      .agg(first("count"))
      .withColumnRenamed("1.0", Configs.PositiveWordsCount)
      .withColumnRenamed("0.0", Configs.NegativeWordsCount)
      .withColumnRenamed("2.0", Configs.NeutralWordsCount)
      .na.fill(0)

    df.join(sentimentCountsDF, Seq("id"), "left").na.fill(0).drop("id")
  }
}
