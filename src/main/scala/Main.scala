import Database.{InitCassandra, InitReviews}
import Libs.{DataAggregation, MachineLearningModel, ModelEnum}

object Main {
  InitCassandra.init()
  if (Configs.SyncCassandraWithDataset) InitReviews.init()

  private val postTrainTestData = Seq(
    ("I love this product!", "positive"),
    ("This is the worst experience ever", "negative"),
    ("I feel neutral about this", "neutral"),
    ("The service was fantastic", "positive"),
    ("I hate waiting in long lines", "negative"),
    ("It's a bad idea", "negative"),
  )

  def main(args: Array[String]): Unit = {
    if (Configs.OnlySyncCassandra) return
    if (Configs.S3LoadModel) {
      MachineLearningModel.inference(postTrainTestData)
      return
    }

    val labeledData = DataAggregation.loadAndPrepareData(SparkInstance.singleton)
    labeledData.show()

    println("Class distribution:")
    labeledData.groupBy("label").count().show()

    val balancedData = DataAggregation.balanceData(labeledData)
    println("Class distribution after balancing:")
    balancedData.groupBy("label").count().show()

    //  MachineLearningModel.trainModel(balancedData, ModelEnum.LogisticRegression, postTrainTestData)
    MachineLearningModel.trainModelWithHyperparameterTuning(
      DataAggregation.weightData(balancedData),
      ModelEnum.LogisticRegression,
      postTrainTestData,
    )

    labeledData.unpersist()
    SparkInstance.singleton.stop()
  }
}