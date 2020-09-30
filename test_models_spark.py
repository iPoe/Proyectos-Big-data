# Initializing a Spark session
from pyspark.sql import SparkSession
import numpy as np
from pyspark.sql.functions import when
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import BinaryClassificationEvaluator



spark = SparkSession.builder.master("local").appName("diabeties").config("spark.some.config.option",
	"some-value").getOrCreate()


raw_data = spark.read.format("csv").option("header","true").option("inferSchema",
 "true").load(r"file:///home/maria_dev/diabetes.csv")




raw_data=raw_data.withColumn("Glucose",
	when(raw_data.Glucose==0,np.nan).otherwise(raw_data.Glucose))

raw_data=raw_data.withColumn("BloodPressure",
	when(raw_data.BloodPressure==0,np.nan).otherwise(raw_data.BloodPressure))

raw_data=raw_data.withColumn("SkinThickness",
	when(raw_data.SkinThickness==0,np.nan).otherwise(raw_data.SkinThickness))

raw_data=raw_data.withColumn("BMI",
	when(raw_data.BMI==0,np.nan).otherwise(raw_data.BMI))

raw_data=raw_data.withColumn("Insulin",
	when(raw_data.Insulin==0,np.nan).otherwise(raw_data.Insulin))


#raw_data.select("Insulin","Glucose","BloodPressure","SkinThickness","BMI").show(5)

imputer=Imputer(inputCols=["Glucose","BloodPressure","SkinThickness","BMI","Insulin"],
	outputCols=["Glucose","BloodPressure","SkinThickness","BMI","Insulin"])

model = imputer.fit(raw_data)

raw_data = model.transform(raw_data)
#raw_data.show(5)


cols=raw_data.columns
cols.remove("Outcome")
# Let us import the vector assembler

assembler = VectorAssembler(inputCols=cols,outputCol="features")
# Now let us use the transform method to transform our dataset
raw_data=assembler.transform(raw_data)
raw_data.select("features").show(truncate=False)



standardscaler=StandardScaler().setInputCol("features").setOutputCol("Scaled_features")
raw_data=standardscaler.fit(raw_data).transform(raw_data)
#raw_data.select("features","Scaled_features").show(5)

train, test = raw_data.randomSplit([0.8, 0.2], seed=12345)

dataset_size=float(train.select("Outcome").count())
numPositives=train.select("Outcome").where('Outcome == 1').count()
per_ones=(float(numPositives)/float(dataset_size))*100
numNegatives=float(dataset_size-numPositives)
#print('The number of ones are {}'.format(numPositives))
#print('Percentage of ones are {}'.format(per_ones))


BalancingRatio= numNegatives/dataset_size

train=train.withColumn("classWeights", 
	when(train.Outcome == 1,BalancingRatio).otherwise(1-BalancingRatio))

css = ChiSqSelector(featuresCol='Scaled_features',outputCol='Aspect',labelCol='Outcome',fpr=0.05)

train=css.fit(train).transform(train)
test=css.fit(test).transform(test)


lr = LogisticRegression(labelCol="Outcome", featuresCol="Aspect",weightCol="classWeights",maxIter=10)
model=lr.fit(train)
predict_train=model.transform(train)
predict_test=model.transform(test)
#predict_test.select("Outcome","prediction").show(10)


#This is the evaluator
evaluator=BinaryClassificationEvaluator(rawPredictionCol="rawPrediction",labelCol="Outcome")
predict_test.select("Outcome","rawPrediction","prediction","probability").show(5)
print("The area under ROC for train set is {}".format(evaluator.evaluate(predict_train)))
print("Esto es accuracy {}".format(evaluator.evaluate(predict_test)))


#Modelo numero 2: DecisionTreeClassifier

from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="Outcome", featuresCol="features")
dt_model = dt.fit(train)
dt_prediction = dt_model.transform(test)

dt_accuracy = evaluator.evaluate(dt_prediction)
print("Accuracy of DecisionTreeClassifier is = %g"% (dt_accuracy))
print("Test Error of DecisionTreeClassifier = %g " % (1.0 - dt_accuracy))



#Naive Bayes 
from pyspark.ml.classification import NaiveBayes
nb = NaiveBayes(labelCol="Outcome",featuresCol="features")
nb_model = nb.fit(train)
nb_prediction = nb_model.transform(test)
nb_accuracy = evaluator.evaluate(nb_prediction)
print("Accuracy of Naive bayes is = %g"%(nb_accuracy))







