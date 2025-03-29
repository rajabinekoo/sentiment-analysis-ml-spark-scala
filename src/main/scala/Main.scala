import Database.{InitCassandra, InitReviews}
import Libs.{DataAggregation, MachineLearningModel, ModelEnum}

object Main {
  InitCassandra.init()
  if (Configs.SyncCassandraWithDataset) InitReviews.init()

  def main(args: Array[String]): Unit = {
    val labeledData = DataAggregation.loadAndPrepareData(SparkInstance.singleton)

    labeledData.show()

    println("Class distribution:")
    labeledData.groupBy("label").count().show()

    val balancedData = DataAggregation.balanceData(labeledData)
    println("Class distribution after balancing:")
    balancedData.groupBy("label").count().show()
    
    val postTrainTestData = Seq(
      ("I love this product!", "positive", 0, 0, 0),
      ("This is the worst experience ever", "negative", 0, 0, 0),
      ("I feel neutral about this", "neutral", 0, 0, 0),
      ("The service was fantastic", "positive", 0, 0, 0),
      ("I hate waiting in long lines", "negative", 0, 0, 0),
      ("Chert o pert", "neutral", 0, 0, 0)
    )
    MachineLearningModel.trainModel(balancedData, ModelEnum.LogisticRegression, postTrainTestData)

    labeledData.unpersist()
    SparkInstance.singleton.stop()
  }
}