import org.apache.spark.sql.SparkSession

package object SparkInstance {
  val singleton: SparkSession = SparkSession.builder()
    .config("spark.driver.memory", Configs.SparkDriverMemory)
    .config("spark.executor.memory", Configs.SparkExecutorMemory)
    .config("spark.sql.shuffle.partitions", Configs.ShufflePartitions)
    .config("spark.cassandra.connection.host", Configs.CassandraHost)
    .config("spark.cassandra.connection.port", Configs.CassandraPort)
    .config("spark.cassandra.input.split.size_in_mb", Configs.PartitioningCassandraSize)
    .config("spark.hadoop.fs.s3a.endpoint", Configs.S3URL)
    .config("spark.hadoop.fs.s3a.access.key", Configs.S3AccessKey)
    .config("spark.hadoop.fs.s3a.secret.key", Configs.S3SecretKey)
    .config("spark.hadoop.fs.s3a.path.style.access", "true")
    .appName(Configs.AppName).master(Configs.Master).getOrCreate()
}
