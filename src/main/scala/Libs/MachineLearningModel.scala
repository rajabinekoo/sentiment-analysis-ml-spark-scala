package Libs

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.ml.{Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.{col, lit, monotonically_increasing_id, when}
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, NGram, StopWordsRemover, Tokenizer, VectorAssembler}

object MachineLearningModel {
  private def definePreprocessingStages(featureSize: Int): Array[PipelineStage with HasInputCol with HasOutputCol with DefaultParamsWritable] = {
    val tokenizer = new Tokenizer().setInputCol("body").setOutputCol("words")
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

  private def evaluateModel(predictions: DataFrame): Double = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = evaluator.evaluate(predictions)
    println(f"Model accuracy is $accuracy")
    accuracy
  }

  private def testWithNewData(model: PipelineModel, data: Seq[(String, String, Int, Int, Int)]): Unit = {
    var testData = SparkInstance.singleton.createDataFrame(data)
      .toDF("body", "label_str", Configs.CoolColumn, Configs.FunnyColumn, Configs.UsefulColumn)
      .withColumn("label",
        when(col("label_str") === "positive", Configs.PositiveLabel)
          .when(col("label_str") === "negative", Configs.NegativeLabel)
          .when(col("label_str") === "neutral", Configs.NeutralLabel)
          .otherwise(lit(null))
      )
      .filter(col("label").isNotNull)
      .withColumn("label", col("label").cast("double"))
      .withColumn("id", monotonically_increasing_id())

    testData = Lexicon.extractSentimentWordsCountFeature(testData)

    val predictions = model.transform(testData)
    println("\nPredictions on new data:")
    predictions.show()
  }

  def trainModel(
    data: DataFrame,
    modelName: String = ModelEnum.LogisticRegression,
    postTrainTestData: Seq[(String, String, Int, Int, Int)] = Seq(),
  ): Unit = {
    val preprocessingStages = definePreprocessingStages(Configs.FeatureSize)

    val classifier = modelName match {
      case ModelEnum.LogisticRegression => new LogisticRegression()
        .setLabelCol("label")
        .setFamily("multinomial")
        .setFeaturesCol("all_features")
        .setMaxIter(Configs.LogisticRegressionIteration)
        .setRegParam(Configs.LearningRateForGradientDescending)
      case ModelEnum.NaiveBayes => new NaiveBayes()
        .setLabelCol("label")
        .setFeaturesCol("all_features")
      case ModelEnum.RandomForest => new RandomForestClassifier()
        .setLabelCol("label")
        .setFeaturesCol("all_features")
        .setNumTrees(Configs.RandomTreeNum)
        .setMaxDepth(Configs.RandomMaxDepth)
        .setSeed(Configs.RandomSeed)
      case _ => throw new Error("Invalid training model")
    }

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

    val pipeline = new Pipeline().setStages(preprocessingStages ++ Array(assembler, classifier))

    val Array(trainingData, testData) = data.randomSplit(
      Array(Configs.TrainingSplitRatio, Configs.TestSplitRatio),
      Configs.RandomSeed,
    )

    val model = pipeline.fit(trainingData)

    val predictions = model.transform(testData)

    evaluateModel(predictions)

    predictions.select(
      "body", "stars", "label", "prediction", "probability",
      Configs.CoolColumn, Configs.FunnyColumn, Configs.UsefulColumn,
    ).limit(Configs.PredictionSampleCount).show()

    if (postTrainTestData.nonEmpty) testWithNewData(model, postTrainTestData)
  }
}
