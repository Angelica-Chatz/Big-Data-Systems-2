import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest

val rdd = sc.textFile("/home/user/heart/processed.cleveland.data")
  
case class MatchData(scores: Array[Double], matched: Boolean) 

def toDouble(s: String) = {
     if ("?".equals(s)) 0.0 else s.toDouble
      }
def parse(line: String) = {
     val pieces = line.split(',')
     val scores = pieces.slice(0,13).map(toDouble)
     val matched =   if (pieces(13)=="0") false else true
         MatchData(scores, matched)
      }

val parsed = rdd.map(parse)

val heartData = parsed.map { md =>
val scores = md.scores
val featureVector = Vectors.dense(scores)
val label = if (md.matched) 1.0 else 0.0
LabeledPoint(label, featureVector)
}


val splits = heartData.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 10
val featureSubsetStrategy = "auto"
val impurity = "gini"
val maxDepth = 4
val maxBins = 32

val model = RandomForest.trainClassifier(trainingData,
numClasses, categoricalFeaturesInfo,
numTrees, featureSubsetStrategy, impurity,
maxDepth, maxBins)

val labelAndPreds = testData.map { point =>
val prediction = model.predict(point.features)
(point.label, prediction)
}

val testErr = labelAndPreds.filter(r => r._1 != r._2).count.toDouble / testData.count()