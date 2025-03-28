package Libs

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.{PipelineModel, PipelineStage}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.{col, lit, monotonically_increasing_id, when}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, NGram, StopWordsRemover, Tokenizer}

object MachineLearningModel {
  def definePreprocessingStages(featureSize: Int): Array[PipelineStage with HasInputCol with HasOutputCol with DefaultParamsWritable] = {
    val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered_words")
    val ngram = new NGram().setInputCol("filtered_words").setOutputCol("ngrams").setN(2)
    val countVectorizer = new CountVectorizer()
      .setInputCol("ngrams")
      .setOutputCol("rawFeatures")
      .setMaxDF(0.9)
      .setMinTF(1)
      .setVocabSize(featureSize)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    Array(tokenizer, remover, ngram, countVectorizer, idf)
  }

  def evaluateModel(predictions: DataFrame): Unit = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(s"Accuracy on test data = $accuracy")
  }

  def testWithNewData(spark: SparkSession, model: PipelineModel, data: Seq[(String, String, Int, Int, Int)]): Unit = {
    var testData = spark.createDataFrame(data)
      .toDF("text", "label_str", Configs.CoolColumn, Configs.FunnyColumn, Configs.UsefulColumn)
      .withColumn("label",
        when(col("label_str") === "positive", Configs.PositiveLabel)
          .when(col("label_str") === "negative", Configs.NegativeLabel)
          .when(col("label_str") === "neutral", Configs.NeutralLabel)
          .otherwise(lit(null))
      )
      .filter(col("label").isNotNull)
      .withColumn("label", col("label").cast("double"))
      .withColumn("review_id", monotonically_increasing_id())

    testData = Lexicon.extractSentimentWordsCountFeature(spark, testData)

    val predictions = model.transform(testData)
    println("\nPredictions on new data:")
    predictions.show()
  }
}
