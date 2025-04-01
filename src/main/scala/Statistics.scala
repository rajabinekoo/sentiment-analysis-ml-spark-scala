import Libs.DataAggregation

object Statistics {
  def main(args: Array[String]): Unit = {
    val labeledData = DataAggregation.loadAndPrepareData(SparkInstance.singleton, 6000000)
    labeledData.groupBy("label").count().show()
  }
}
