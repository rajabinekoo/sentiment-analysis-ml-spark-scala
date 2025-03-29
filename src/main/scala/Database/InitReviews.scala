package Database

import Libs.Lexicon
import org.apache.spark.sql.functions._

object InitReviews {
  def init(): Unit = {
    var df = SparkInstance.singleton.read
      .format("json")
      .option("header", "true")
      .option("inferSchema", "true")
      .load(Configs.DataSource)
      .limit(Configs.ExampleLimit)
      .filter(col("stars").isNotNull)
      .withColumnRenamed("text", "body")
      .withColumn("label",
        when(col("stars") >= Configs.PositiveThreshold, Configs.PositiveLabel)
          .when(col("stars") <= Configs.NegativeThreshold, Configs.NegativeLabel)
          .when(col("stars") === Configs.NeutralThreshold, Configs.NeutralLabel)
          .otherwise(lit(null))
      )
      .filter(col("label").isNotNull)
      .withColumn("label", col("label").cast("double"))
      .withColumn("date", to_timestamp(col("date"), "yyyy-MM-dd HH:mm:ss"))
      .filter(col(Configs.CoolColumn).isNotNull)
      .filter(col(Configs.FunnyColumn).isNotNull)
      .filter(col(Configs.UsefulColumn).isNotNull)
      .withColumn(Configs.CoolColumn, col(Configs.CoolColumn).cast("long"))
      .withColumn(Configs.FunnyColumn, col(Configs.FunnyColumn).cast("long"))
      .withColumn(Configs.UsefulColumn, col(Configs.UsefulColumn).cast("long"))
      .withColumn("id", monotonically_increasing_id())

    df = Lexicon.extractSentimentWordsCountFeature(df)

    df.show()

    df.select(
        col("body"), col("review_id"), col("date"), col("business_id"), col("user_id"), col("stars").cast("int"),
        col(Configs.CoolColumn), col(Configs.FunnyColumn), col(Configs.UsefulColumn), col("label"),
        col(Configs.PositiveWordsCount), col(Configs.NegativeWordsCount), col(Configs.NeutralWordsCount)
      ).write.format("org.apache.spark.sql.cassandra")
      .option("keyspace", Configs.CassandraKeyspace)
      .option("table", Configs.BucketedReviewsTable)
      .mode("append").save()
  }
}
