package Database

import com.datastax.spark.connector.cql.CassandraConnector

object InitCassandra {
  def init(): Unit = {
    val connector = CassandraConnector(SparkInstance.singleton.sparkContext.getConf)

    connector.withSessionDo { session =>
      session.execute(
        s"""
          CREATE KEYSPACE IF NOT EXISTS ${Configs.CassandraKeyspace}
          WITH REPLICATION = {'class': 'SimpleStrategy', 'replication_factor': 1};
        """)

      session.execute(s"USE ${Configs.CassandraKeyspace};")

      session.execute(
        s"""
          CREATE TABLE IF NOT EXISTS ${Configs.BucketedReviewsTable} (
            review_id TEXT,
            body TEXT,
            label FLOAT,
            user_id TEXT,
            stars BIGINT,
            date TIMESTAMP,
            business_id TEXT,
            ${Configs.NeutralWordsCount} BIGINT,
            ${Configs.PositiveWordsCount} BIGINT,
            ${Configs.NegativeWordsCount} BIGINT,
            PRIMARY KEY ((review_id), date, business_id)
          ) WITH CLUSTERING ORDER BY (date DESC);
        """)

      session.close()
    }
  }
}
