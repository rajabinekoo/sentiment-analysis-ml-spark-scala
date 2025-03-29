package Libs

import org.apache.spark.sql.functions._
import org.apache.spark.storage.StorageLevel
import org.apache.spark.sql.{DataFrame, SparkSession}

object DataAggregation {
  def loadAndPrepareData(spark: SparkSession): DataFrame = {
    spark.read
      .format("org.apache.spark.sql.cassandra")
      .option("keyspace", Configs.CassandraKeyspace)
      .option("table", Configs.BucketedReviewsTable)
      .load()
      .limit(Configs.ExampleLimit)
      .persist(Configs.DataframePersistLevel)
  }

  def balanceData(data: DataFrame): DataFrame = {
    val neutralData = data.filter(col("label") === Configs.NeutralLabel)
    val positiveData = data.filter(col("label") === Configs.PositiveLabel)
    val negativeData = data.filter(col("label") === Configs.NegativeLabel)

    val neutralCount = neutralData.count()
    val positiveCount = positiveData.count()
    val negativeCount = negativeData.count()

    val maxCount = Seq(positiveCount, negativeCount, neutralCount).max

    val balancedNeutral = if (neutralCount < maxCount) neutralData.sample(withReplacement = true, fraction = (maxCount.toDouble / neutralCount)) else neutralData
    val balancedPositive = if (positiveCount < maxCount) positiveData.sample(withReplacement = true, fraction = (maxCount.toDouble / positiveCount)) else positiveData
    val balancedNegative = if (negativeCount < maxCount) negativeData.sample(withReplacement = true, fraction = (maxCount.toDouble / negativeCount)) else negativeData

    balancedPositive.unionByName(balancedNegative).unionByName(balancedNeutral)
  }
}
