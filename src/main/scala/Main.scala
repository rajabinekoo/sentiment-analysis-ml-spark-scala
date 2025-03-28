import Libs.{DataAggregation, Lexicon, MachineLearningModel}

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.{Pipeline, PipelineStage}
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.sql.functions.{col, monotonically_increasing_id}
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, RandomForestClassifier}

object Main {
  private val spark = SparkSession.builder().appName(Configs.AppName).master(Configs.Master).getOrCreate()
  Lexicon.init(spark)

  def main(args: Array[String]): Unit = {
    var labeledData = DataAggregation.loadAndPrepareData(spark, Configs.DataSource, Configs.ExampleLimit)
      .filter(col(Configs.CoolColumn).isNotNull)
      .filter(col(Configs.FunnyColumn).isNotNull)
      .filter(col(Configs.UsefulColumn).isNotNull)
      .withColumn("review_id", monotonically_increasing_id())
      .withColumn(Configs.CoolColumn, col(Configs.CoolColumn).cast("long"))
      .withColumn(Configs.FunnyColumn, col(Configs.FunnyColumn).cast("long"))
      .withColumn(Configs.UsefulColumn, col(Configs.UsefulColumn).cast("long"))

    labeledData = Lexicon.extractSentimentWordsCountFeature(spark, labeledData)

    println("Class distribution:")
    labeledData.groupBy("label").count().show()

    val balancedData = DataAggregation.balanceData(labeledData)
    println("Class distribution after balancing:")
    balancedData.groupBy("label").count().show()

    val preprocessingStages = MachineLearningModel.definePreprocessingStages(Configs.FeatureSize)

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "features",
        Configs.CoolColumn,
        Configs.FunnyColumn,
        Configs.UsefulColumn,
        Configs.NeutralWordsCount,
        Configs.PositiveWordsCount,
        Configs.NegativeWordsCount
      ))
      .setOutputCol("all_features")

    trainLogisticRegression(preprocessingStages, assembler, balancedData)

    spark.stop()
  }

  private def trainLogisticRegression(
    preprocessingStages: Array[PipelineStage with HasInputCol with HasOutputCol with DefaultParamsWritable],
    assembler: VectorAssembler,
    balancedData: DataFrame,
  ): Unit = {
    val classifier = new LogisticRegression()
      .setLabelCol("label")
      .setFeaturesCol("all_features")
      .setMaxIter(100)
      .setFamily("multinomial")

    val pipeline = new Pipeline().setStages(preprocessingStages ++ Array(assembler, classifier))

    val Array(trainingData, testData) = balancedData.randomSplit(Array(Configs.TrainingSplitRatio, Configs.TestSplitRatio), seed = Configs.RandomSeed)

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    MachineLearningModel.evaluateModel(predictions)

    predictions.select(
      "text", "stars", "label", "prediction", "probability",
      Configs.CoolColumn, Configs.FunnyColumn, Configs.UsefulColumn,
    ).limit(Configs.PredictionSampleCount).show()

    val postTrainTestData = Seq(
      ("I love this product!", "positive", 0, 0, 0),
      ("This is the worst experience ever", "negative", 0, 0, 0),
      ("I feel neutral about this", "neutral", 0, 0, 0),
      ("The service was fantastic", "positive", 0, 0, 0),
      ("I hate waiting in long lines", "negative", 0, 0, 0),
      ("Chert o pert", "neutral", 0, 0, 0)
    )
    MachineLearningModel.testWithNewData(spark, model, postTrainTestData)
  }

  private def trainRandomForest(
    preprocessingStages: Array[PipelineStage with HasInputCol with HasOutputCol with DefaultParamsWritable],
    assembler: VectorAssembler,
    balancedData: DataFrame,
  ): Unit = {
    val classifier = new RandomForestClassifier()
      .setLabelCol("label")
      .setFeaturesCol("all_features")
      .setNumTrees(300)
      .setMaxDepth(10)
      .setSeed(Configs.RandomSeed)

    val pipeline = new Pipeline().setStages(preprocessingStages ++ Array(assembler, classifier))

    val Array(trainingData, testData) = balancedData.randomSplit(Array(Configs.TrainingSplitRatio, Configs.TestSplitRatio), seed = Configs.RandomSeed)

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    MachineLearningModel.evaluateModel(predictions)

    predictions.select(
      "text", "stars", "label", "prediction", "probability",
      Configs.CoolColumn, Configs.FunnyColumn, Configs.UsefulColumn,
    ).limit(Configs.PredictionSampleCount).show()

    val postTrainTestData = Seq(
      ("I love this product!", "positive", 0, 0, 0),
      ("This is the worst experience ever", "negative", 0, 0, 0),
      ("I feel neutral about this", "neutral", 0, 0, 0),
      ("The service was fantastic", "positive", 0, 0, 0),
      ("I hate waiting in long lines", "negative", 0, 0, 0),
      ("Chert o pert", "neutral", 0, 10, 0)
    )
    MachineLearningModel.testWithNewData(spark, model, postTrainTestData)
  }

  private def trainNaiveBayes(
    preprocessingStages: Array[PipelineStage with HasInputCol with HasOutputCol with DefaultParamsWritable],
    assembler: VectorAssembler,
    balancedData: DataFrame,
  ): Unit = {
    val classifier = new NaiveBayes()
      .setLabelCol("label")
      .setFeaturesCol("all_features")

    val pipeline = new Pipeline().setStages(preprocessingStages ++ Array(assembler, classifier))

    val Array(trainingData, testData) = balancedData.randomSplit(Array(Configs.TrainingSplitRatio, Configs.TestSplitRatio), seed = Configs.RandomSeed)

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    MachineLearningModel.evaluateModel(predictions)

    predictions.select(
      "text", "stars", "label", "prediction", "probability",
      Configs.CoolColumn, Configs.FunnyColumn, Configs.UsefulColumn,
    ).limit(Configs.PredictionSampleCount).show()

    val postTrainTestData = Seq(
      ("I love this product!", "positive", 0, 0, 0),
      ("This is the worst experience ever", "negative", 0, 0, 0),
      ("I feel neutral about this", "neutral", 0, 0, 0),
      ("The service was fantastic", "positive", 0, 0, 0),
      ("I hate waiting in long lines", "negative", 0, 0, 0),
      ("Chert o pert", "neutral", 0, 10, 0)
    )
    MachineLearningModel.testWithNewData(spark, model, postTrainTestData)
  }
}