import org.apache.spark.storage.StorageLevel

package object Configs {
  val AppName = "SentimentAnalysisYelpImprovedLogisticRegression"
  val Master = "local[*]"
  val DataSource = "src/main/resources/yelp.json"
  val ExampleLimit = 20000
  val FeatureSize = 20000
  val TrainingSplitRatio = 0.8
  val TestSplitRatio = 0.2
  val RandomSeed = 12345
  val PositiveThreshold = 4
  val NegativeThreshold = 2
  val NeutralThreshold = 3
  val PositiveLabel = 1.0
  val NegativeLabel = 0.0
  val NeutralLabel = 2.0
  val PredictionSampleCount = 5
  val CoolColumn = "cool"
  val FunnyColumn = "funny"
  val UsefulColumn = "useful"
  val NeutralWordsCount = "neutral_word_counts"
  val PositiveWordsCount = "positive_word_counts"
  val NegativeWordsCount = "negative_word_counts"
  val VaderLexiconPath = "src/main/resources/vader_lexicon.txt"
  val BucketedReviewsTable = "reviews_bucketed"
  val CassandraHost = "localhost"
  val CassandraPort = "9042"
  val CassandraKeyspace = "yelp_data_mining"
  val SyncCassandraWithDataset = false
  val PartitioningCassandraSize = "64" // 64mb
  val LearningRateForGradientDescending = 0.0001
  val ShufflePartitions = "200"
  val SparkDriverMemory = "1g"
  val SparkExecutorMemory = "1g"
  val DataframePersistLevel: StorageLevel = StorageLevel.DISK_ONLY
  val LogisticRegressionIteration = 100
  val S3AccessKey = "hg536gS5bVdByztLJjJv"
  val S3SecretKey = "6FwhSehsU4YDHUInsSOGe8wFGtH2mUJcmpNWisrt"
  val S3URL = "http://127.0.0.1:9000"
  val ModelPath = "s3a://machine-learning-models/spark-model/"
}
