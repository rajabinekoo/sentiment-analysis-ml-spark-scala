package Libs

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.{col, lit, when}

object DataAggregation {
  def loadAndPrepareData(spark: SparkSession, dataSource: String, limit: Int = Int.MaxValue): DataFrame = {
    spark.read
      .format("json")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(dataSource)
      .limit(limit)
      .filter(col("stars").isNotNull)
      .withColumn("label",
        when(col("stars") >= Configs.PositiveThreshold, Configs.PositiveLabel)
          .when(col("stars") <= Configs.NegativeThreshold, Configs.NegativeLabel)
          .when(col("stars") === Configs.NeutralThreshold, Configs.NeutralLabel) // اضافه کردن برچسب خنثی
          .otherwise(lit(null))
      )
      .filter(col("label").isNotNull)
      .withColumn("label", col("label").cast("double")).limit(2000)
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
