package Libs

import org.apache.spark.sql.functions._
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataAggregation {
  def loadAndPrepareData(spark: SparkSession, limit: Int = Configs.LoadExampleLimit): DataFrame = {
    spark.read
      .format("org.apache.spark.sql.cassandra")
      .option("keyspace", Configs.CassandraKeyspace)
      .option("table", Configs.BucketedReviewsTable)
      .load()
      .limit(limit)
      .persist(Configs.DataframePersistLevel)
  }

  def balanceData(data: DataFrame, balancingFactor: Double = 0.8): DataFrame = {
    val neutralData = data.filter(col("label") === Configs.NeutralLabel)
    val positiveData = data.filter(col("label") === Configs.PositiveLabel)
    val negativeData = data.filter(col("label") === Configs.NegativeLabel)

    val neutralCount = neutralData.count()
    val positiveCount = positiveData.count()
    val negativeCount = negativeData.count()

    val maxCount = Seq(positiveCount, negativeCount, neutralCount).max
    val targetCount = (maxCount * balancingFactor).toInt

    val balancedNeutral = if (neutralCount < targetCount) neutralData.sample(withReplacement = true, fraction = (targetCount.toDouble / neutralCount)) else neutralData
    val balancedPositive = if (positiveCount < targetCount) positiveData.sample(withReplacement = true, fraction = (targetCount.toDouble / positiveCount)) else positiveData
    val balancedNegative = if (negativeCount < targetCount) negativeData.sample(withReplacement = true, fraction = (targetCount.toDouble / negativeCount)) else negativeData

    balancedPositive.unionByName(balancedNegative).unionByName(balancedNeutral)
  }
  
  def weightData(data: DataFrame): DataFrame = {
    val positiveCount = data.filter(col("label") === Configs.PositiveLabel).count()
    val negativeCount = data.filter(col("label") === Configs.NegativeLabel).count()
    val neutralCount = data.filter(col("label") === Configs.NeutralLabel).count()
    val totalCount = data.count().toDouble

    val positiveWeight = totalCount / (positiveCount + 1e-6)
    val negativeWeight = totalCount / (negativeCount + 1e-6)
    val neutralWeight = totalCount / (neutralCount + 1e-6)

    data.withColumn(
      "classWeight",
      when(col("label") === Configs.PositiveLabel, positiveWeight)
        .when(col("label") === Configs.NegativeLabel, negativeWeight)
        .otherwise(neutralWeight)
    )
  }
}
