package Libs

import java.io.File
import java.nio.file.Paths

import org.apache.spark.sql.DataFrame
import org.apache.spark.ml.util.DefaultParamsWritable
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.param.shared.{HasInputCol, HasOutputCol}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.{Model, Pipeline, PipelineModel, PipelineStage}
import org.apache.spark.sql.functions.{col, lit, monotonically_increasing_id, when}
import org.apache.spark.ml.classification.{LogisticRegression, NaiveBayes, RandomForestClassifier}
import org.apache.spark.ml.feature.{CountVectorizer, IDF, NGram, StopWordsRemover, VectorAssembler, Word2Vec, Tokenizer => MLTokenizer}

object MachineLearningModel {
  private def definePreprocessingStages(featureSize: Int): Array[PipelineStage with HasInputCol with HasOutputCol with DefaultParamsWritable] = {
    val tokenizer = new MLTokenizer().setInputCol("body").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered_words")
    val ngram = new NGram().setInputCol("filtered_words").setOutputCol("ngrams").setN(1)
    val countVectorizer = new CountVectorizer()
      .setInputCol("ngrams")
      .setOutputCol("rawFeatures")
      .setMaxDF(0.9)
      .setMinTF(1)
      .setVocabSize(featureSize)
    val idf = new IDF().setInputCol("rawFeatures").setOutputCol("features")
    Array(tokenizer, remover, ngram, countVectorizer, idf)
  }

  private def definePreprocessingStages2(featureSize: Int): Array[PipelineStage with HasInputCol with HasOutputCol with DefaultParamsWritable] = {
    val tokenizer = new MLTokenizer().setInputCol("body").setOutputCol("words")
    val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered_words")
    val ngram = new NGram().setInputCol("filtered_words").setOutputCol("ngrams").setN(1)
    val word2Vec = new Word2Vec()
      .setInputCol("ngrams")
      .setOutputCol("features")
      .setVectorSize(featureSize)
      .setMinCount(1)
    Array(tokenizer, remover, ngram, word2Vec)
  }


  private def evaluateModel(predictions: DataFrame): Double = {
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val f1Score = evaluator.evaluate(predictions)
    println(f"Model F1 score is $f1Score")

    val accuracyEvaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    val accuracy = accuracyEvaluator.evaluate(predictions)
    println(f"Model accuracy is $accuracy")
    accuracy
  }

  private def calculateConfusionMatrix(predictions: DataFrame): Unit = {
    val predictionAndLabels = predictions.select(col("prediction"), col("label").cast("double"))
      .rdd.map(row => (row.getDouble(0), row.getDouble(1)))

    val metrics = new MulticlassMetrics(predictionAndLabels)

    val confusionMatrix = metrics.confusionMatrix

    println("  \tPredicted 0\tPredicted 1\tPredicted 2")
    println(s"Actual 0\t${confusionMatrix(0, 0)}\t${confusionMatrix(0, 1)}\t${confusionMatrix(0, 2)}")
    println(s"Actual 1\t${confusionMatrix(1, 0)}\t${confusionMatrix(1, 1)}\t${confusionMatrix(1, 2)}")
    println(s"Actual 2\t${confusionMatrix(2, 0)}\t${confusionMatrix(2, 1)}\t${confusionMatrix(2, 2)}")
  }

  private def testWithNewData(model: Model[_], data: Seq[(String, String)]): Unit = {
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

    calculateConfusionMatrix(predictions)

    if (postTrainTestData.nonEmpty) testWithNewData(model, postTrainTestData)
  }

  def trainModelWithHyperparameterTuning(
    data: DataFrame,
    modelName: String = ModelEnum.LogisticRegression,
    postTrainTestData: Seq[(String, String)] = Seq(),
    featureSize: Int = Configs.FeatureSize,
  ): Unit = {
    val preprocessingStages = definePreprocessingStages(featureSize)
    val classifier = getModelByName(modelName).asInstanceOf[LogisticRegression] // Cast به LogisticRegression

    val assembler = new VectorAssembler()
      .setInputCols(Array(
        "features",
        Configs.NeutralWordsCount,
        Configs.PositiveWordsCount,
        Configs.NegativeWordsCount
      ))
      .setOutputCol("all_features")

    val pipeline = new Pipeline().setStages(preprocessingStages ++ Array(assembler, classifier.setWeightCol("classWeight")))

    val Array(trainingData, testData) = data.randomSplit(
      Array(Configs.TrainingSplitRatio, Configs.TestSplitRatio),
      Configs.RandomSeed,
    )

    val paramGrid = new ParamGridBuilder()
      .addGrid(classifier.regParam, Array(0.01, 0.1, 1.0))
      .addGrid(classifier.elasticNetParam, Array(0.0, 0.5, 1.0))
      .addGrid(classifier.maxIter, Array(100, 200))
      .build()

    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol("label")
      .setPredictionCol("prediction")
      .setMetricName("f1")

    val cv = new CrossValidator()
      .setEstimator(pipeline)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setNumFolds(3)

    val cvModel = cv.fit(trainingData)

    val bestModel = cvModel.bestModel

    val predictions = bestModel.transform(testData)

    evaluateModel(predictions)
    calculateConfusionMatrix(predictions)

    predictions.select(
      "body", "stars", "label", "prediction", "probability",
    ).limit(Configs.PredictionSampleCount).show()

    if (postTrainTestData.nonEmpty) testWithNewData(bestModel, postTrainTestData)
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
