package Libs

import java.io.File
import java.nio.file.Paths

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

  private def testWithNewData(model: PipelineModel, data: Seq[(String, String)]): Unit = {
    var testData = SparkInstance.singleton.createDataFrame(data)
      .toDF("body", "label_str")
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

  private def getModelByName(modelName: String): org.apache.spark.ml.classification.Classifier[_, _, _] = {
    modelName match {
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
  }

  def trainModel(
    data: DataFrame,
    modelName: String = ModelEnum.LogisticRegression,
    postTrainTestData: Seq[(String, String)] = Seq(),
  ): Unit = {
    val preprocessingStages = definePreprocessingStages(Configs.FeatureSize)
    val classifier = getModelByName(modelName)

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "features",
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

    val accuracy = evaluateModel(predictions)
    if (accuracy > 0.9 && Configs.S3SaveModel) {
      if (new File(Configs.S3LocalModelPath).isDirectory) {
        DiskStorage.deleteDirectoryRecursively(Paths.get(Configs.S3LocalModelPath))
      }
      model.save(Configs.SaveModelPath)
      S3Storage.uploadDirectory(Configs.S3LocalModelPath, Configs.S3ModelBucket, Configs.S3RemoteModelPath)
    }

    predictions.select(
      "body", "stars", "label", "prediction", "probability",
    ).limit(Configs.PredictionSampleCount).show()

    if (postTrainTestData.nonEmpty) testWithNewData(model, postTrainTestData)
  }

  def inference(postTrainTestData: Seq[(String, String)]): Unit = {
    if (!Configs.S3LoadModel) return
    val localDir = new File(Configs.S3LocalModelPath)
    if (!localDir.isDirectory) {
      S3Storage.downloadDirectory(Configs.S3ModelBucket, Configs.S3RemoteModelPath, Configs.S3LocalModelPath)
    }
    val model = PipelineModel.load(Configs.S3LocalModelPath)
    testWithNewData(model, postTrainTestData)
  }
}
