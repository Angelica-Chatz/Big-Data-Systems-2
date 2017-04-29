rf-regression.scala
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.regression.LabeledPoint
import org.apache.spark.mllib.tree.RandomForest

val rdd = sc.textFile("/home/user/spark/housing/housing.data")
  
case class MatchData(scores: Array[Double], matched: Double) 

def toDouble(s: String) = {
     if ("?".equals(s)) 0.0 else s.toDouble
      }
def parse(line: String) = {
     val pieces = line.trim.split("\\s+")
     val scores = pieces.slice(0,13).map(toDouble)
     val matched =   pieces(13).toDouble
         MatchData(scores, matched)
      }


val parsed = rdd.map(parse)
parsed.first()

val housingData = parsed.map { md =>
val scores = md.scores
val featureVector = Vectors.dense(scores)
val label = md.matched
LabeledPoint(label, featureVector)
}


val splits = housingData.randomSplit(Array(0.7, 0.3))
val (trainingData, testData) = (splits(0), splits(1))

val numClasses = 2
val categoricalFeaturesInfo = Map[Int, Int]()
val numTrees = 10
val featureSubsetStrategy = "auto"
val impurity = "variance"
val maxDepth = 4
val maxBins = 32

val model = RandomForest.trainRegressor(trainingData, categoricalFeaturesInfo, numTrees, featureSubsetStrategy, impurity,
maxDepth, maxBins)

val labelAndPreds = testData.map { point =>
val prediction = model.predict(point.features)
(point.label, prediction)
}

val testMSE = labelAndPreds.map{ case(v, p) => math.pow((v - p), 2)}.mean()
println("Test Mean Squared Error = " + testMSE)