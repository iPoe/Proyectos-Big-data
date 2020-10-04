import pyspark.sql.functions as F
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col, explode, array, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import ChiSqSelector
from pyspark.mllib.evaluation import MulticlassMetrics
from pyspark.sql.types import FloatType
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import LogisticRegression,OneVsRest
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import dayofweek
from pyspark.sql.functions import isnan, when, count, col, lit, sum
from pyspark.sql.functions import (to_date, datediff, date_format,month)
#Este es un comentario
#Se carga el conjunto de datos

spark = SparkSession.builder.master("local").appName("Avila").config("spark.some.config.option","some-value").getOrCreate()
data = spark.read.format("csv").option("header","true").option("inferSchema", "true").load(r"avila.csv")

#####################################################################################################
#PRIMER PUNTO
#DESCRIPCION DEL CONJUNTO DE DATOS INICIAL
#Imprime la cantidad de registros y atributos respectivamente
"""
print("Registros Iniciales:",data.count(),", Atributos Iniciales:",len(data.columns))

#Tipo de los atributos
data.printSchema()

# Se revisa si existen nulos en alguno de los atributos del dataset
print("Cantidad de Nulos en cada atributo")
print(data.select([count(when(isnan(c),c)).alias(c) for c in data.columns]).toPandas().head())

#Descripcion de los atributos
print("Descripcion de los atributos")
print(data.describe().select("Summary","F1","F2","F3","F4","F5").show())
print(data.describe().select("Summary","F6","F7","F8","F9","F10").show())

#Se verifica la correlacion entre los atributos
pd = data.toPandas()
print("Correlacion entre atributos")
print(pd.corr())

#Diagrama de cajas para verificar datos atipicos
#plt.boxplot((pd['F1'],pd['F2'],pd['F3'],pd['F4'],pd['F5'],pd['F6'],pd['F7'],pd['F8'],pd['F9'],pd['F10']))
#plt.show()

#Distribucion del atributo clasificador
print("Distribucion del atributo clasificador")
data.groupby("Author").count().show()

"""
#####################################################################################################
#COMIENZA EL SEGUNDO PUNTO
#LIMPIEZA DE LOS DATOS

#Como se puede ver en los diagramas de cajas, el atributo F2 tiene datos que son demasiado atipicos
#Estos registros se eliminaran
#print("LIMPIEZA DE LOS DATOS")
data = data.filter(data.F2<350)
#print("Datos Demasiado Atipicos de F2 Eliminados:",data.count())

#Se elimina el atributo F10
data = data.drop('F6')
#data = data.drop('F10')
#print("Atributo F10 Eliminado:",data.columns)

#print("Conversion de atributos categoricos a numericos")
indexer = StringIndexer(inputCol="Author", outputCol="AuthorNum")
data = indexer.fit(data).transform(data)
data = data.drop('Author')
data.groupby("AuthorNum").count().show()

#Prueba con df sin balancear
raw_data = data

#Se balancea cada categoria
#A.groupby("AuthorNum").count().show()
A = data.filter(data.AuthorNum == 0.0).sample(fraction=0.35)

F = data.filter(col("AuthorNum") == 1.0).withColumn("dummy", explode(array([lit(x) for x in range(1)]))).drop('dummy')
E = data.filter(col("AuthorNum") == 2.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
I = data.filter(col("AuthorNum") == 3.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
X = data.filter(col("AuthorNum") == 4.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
H = data.filter(col("AuthorNum") == 5.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
G = data.filter(col("AuthorNum") == 6.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
D = data.filter(col("AuthorNum") == 7.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
Y = data.filter(col("AuthorNum") == 8.0).withColumn("dummy", explode(array([lit(x) for x in range(4)]))).drop('dummy')
C = data.filter(col("AuthorNum") == 9.0).withColumn("dummy", explode(array([lit(x) for x in range(5)]))).drop('dummy')
W = data.filter(col("AuthorNum") == 10.0).withColumn("dummy", explode(array([lit(x) for x in range(5)]))).drop('dummy')
B = data.filter(col("AuthorNum") == 11.0).withColumn("dummy", explode(array([lit(x) for x in range(100)]))).drop('dummy')
# F = data.filter(col("AuthorNum") == 1.0).withColumn("dummy", explode(array([lit(x) for x in range(1)]))).drop('dummy')
# E = data.filter(col("AuthorNum") == 2.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
# I = data.filter(col("AuthorNum") == 3.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
# X = data.filter(col("AuthorNum") == 4.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
# H = data.filter(col("AuthorNum") == 5.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
# G = data.filter(col("AuthorNum") == 6.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
# D = data.filter(col("AuthorNum") == 7.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
# Y = data.filter(col("AuthorNum") == 8.0).withColumn("dummy", explode(array([lit(x) for x in range(4)]))).drop('dummy')
# C = data.filter(col("AuthorNum") == 9.0).withColumn("dummy", explode(array([lit(x) for x in range(7)]))).drop('dummy')
# W = data.filter(col("AuthorNum") == 10.0).withColumn("dummy", explode(array([lit(x) for x in range(14)]))).drop('dummy')
# B = data.filter(col("AuthorNum") == 11.0).withColumn("dummy", explode(array([lit(x) for x in range(100)]))).drop('dummy')

#Se juntan todas las categorias balanceadas
data = A.union(B).union(C).union(D).union(E).union(F).union(G).union(H).union(I).union(W).union(Y).union(X)
#data = A.union(C).union(D).union(E).union(F).union(G).union(H).union(I).union(W).union(Y).union(X)
print("Conjunto Balanceado")
data.groupby("AuthorNum").count().show()
print("Numero de Registros Dataset Limpio:",data.count(),", Atributos:",len(data.columns))

#####################################################################################################
#COMIENZA PUNTO 3
#Entrenamiento de modelos:
#Modelo 1

cols=data.columns
cols.remove("AuthorNum")
# Let us import the vector assembler

assembler = VectorAssembler(inputCols=cols,outputCol="features")
# Now let us use the transform method to transform our dataset
data=assembler.transform(data)
#data.select("features").show(truncate=False)
train, test = data.randomSplit([0.8, 0.2],seed=20)

raw_data=assembler.transform(raw_data)
train2, test2 = raw_data.randomSplit([0.8, 0.2])

lr = LogisticRegression(labelCol="AuthorNum",maxIter=50,featuresCol="features",family="multinomial",elasticNetParam=0.8,regParam=0.1)



# Fit the model
lrModel = lr.fit(train)

predict_test=lrModel.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="AuthorNum",
	predictionCol="prediction", metricName="accuracy")

lr_accuracy = evaluator.evaluate(predict_test)
print("Accuracy score of LogisticRegression is = %g"% (lr_accuracy))
#Matriz de confusion



#Modelo 2

from pyspark.ml.classification import DecisionTreeClassifier



dt = DecisionTreeClassifier(labelCol="AuthorNum", featuresCol="features",maxDepth=20)
dt_model = dt.fit(train)
dt_prediction = dt_model.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="AuthorNum",
                                              predictionCol="prediction", metricName="accuracy")

dt_accuracy = evaluator.evaluate(dt_prediction)
print("Accuracy Score of DecisionTreeClassifier is = %g"% (dt_accuracy))

preds_and_labels = dt_prediction.select(['prediction','AuthorNum']).withColumn('label', F.col('AuthorNum').cast(FloatType())).orderBy('prediction')
preds_and_labels = preds_and_labels.select(['prediction','AuthorNum'])
tp = preds_and_labels.rdd.map(tuple)
metrics = MulticlassMetrics(tp)
print(metrics.confusionMatrix().toArray())




#Random forest

from pyspark.ml.classification import RandomForestClassifier
rf = RandomForestClassifier(labelCol="AuthorNum", featuresCol="features",numTrees=10,subsamplingRate=1,maxDepth=10)
rf_model = rf.fit(train)
rf_prediction = rf_model.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="AuthorNum",
                                              predictionCol="prediction", metricName="accuracy")

rf_accuracy = evaluator.evaluate(rf_prediction)
print("Accuracy Score of RandomForestClassifier is = %g"% (rf_accuracy))


#Modelo LSVC
# svm = LinearSVC(maxIter=50,regParam=0.1)
# ovr = OneVsRest(classifier=svm,featuresCol="features",labelCol="AuthorNum")
# ovrModel = ovr.fit(train)

# evaluator = MulticlassClassificationEvaluator(metricName="f1",labelCol="AuthorNum",predictionCol="prediction")

# predictions = ovrModel.transform(test)

# print("Accuracy: {}".format(evaluator.evaluate(predictions)))