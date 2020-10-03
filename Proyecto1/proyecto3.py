from pyspark.sql import SparkSession
from pyspark.sql.functions import isnan, when, count, col, explode, array, lit
from pyspark.ml.feature import StringIndexer
from pyspark.ml.feature import Imputer
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import ChiSqSelector
from pyspark.ml.classification import LogisticRegression
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
print("Registros:",data.count(),", Atributos:",len(data.columns))

#Tipo de los atributos
data.printSchema()

# Se revisa si existen nulos en alguno de los atributos del dataset
print(data.select([count(when(isnan(c),c)).alias(c) for c in data.columns]).toPandas().head())

#Descripcion de los atributos
print(data.describe().select("Summary","F1","F2","F3","F4","F5").show())
print(data.describe().select("Summary","F6","F7","F8","F9","F10").show())

#Se verifica la correlacion entre los atributos
pd = data.toPandas()
print("CORRELATION")
print(pd.corr())

#Diagrama de cajas para verificar datos atipicos
#plt.boxplot((pd['F1'],pd['F2'],pd['F3'],pd['F4'],pd['F5'],pd['F6'],pd['F7'],pd['F8'],pd['F9'],pd['F10']))
#plt.show()


#Distribucion del atributo clasificador
data.groupby("Author").count().show()

"""
#####################################################################################################
#COMIENZA EL SEGUNDO PUNTO
#LIMPIEZA DE LOS DATOS

#Como se puede ver en los diagramas de cajas, el atributo F2 tiene datos que son demasiado atipicos
#Estos registros se eliminaran
#print("LIMPIEZA DE LOS DATOS")
data = data.filter(data.F2<350)
#print("Datos Atipicos F2 Eliminados:",data.count())

#Se elimina el atributo F10
data = data.drop('F10')
#print("Atributo F10 Eliminado:",data.columns)

#Convertir los atributos categoricos a numericos
indexer = StringIndexer(inputCol="Author", outputCol="AuthorNum")
data = indexer.fit(data).transform(data)
data = data.drop('Author')
#data.groupby("AuthorNum").count().show()

#Se balancea cada categoria
A = data.filter(data.AuthorNum == 0.0).sample(fraction=0.9)
#A.groupby("AuthorNum").count().show()
F = data.filter(col("AuthorNum") == 1.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
E = data.filter(col("AuthorNum") == 2.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
I = data.filter(col("AuthorNum") == 3.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
X = data.filter(col("AuthorNum") == 4.0).withColumn("dummy", explode(array([lit(x) for x in range(5)]))).drop('dummy')
H = data.filter(col("AuthorNum") == 5.0).withColumn("dummy", explode(array([lit(x) for x in range(5)]))).drop('dummy')
G = data.filter(col("AuthorNum") == 6.0).withColumn("dummy", explode(array([lit(x) for x in range(6)]))).drop('dummy')
D = data.filter(col("AuthorNum") == 7.0).withColumn("dummy", explode(array([lit(x) for x in range(7)]))).drop('dummy')
Y = data.filter(col("AuthorNum") == 8.0).withColumn("dummy", explode(array([lit(x) for x in range(10)]))).drop('dummy')
C = data.filter(col("AuthorNum") == 9.0).withColumn("dummy", explode(array([lit(x) for x in range(10)]))).drop('dummy')
W = data.filter(col("AuthorNum") == 10.0).withColumn("dummy", explode(array([lit(x) for x in range(10)]))).drop('dummy')
B = data.filter(col("AuthorNum") == 11.0).withColumn("dummy", explode(array([lit(x) for x in range(10)]))).drop('dummy')

#Se juntan todas las categorias balanceadas
data = A.union(B).union(C).union(D).union(E).union(F).union(G).union(H).union(I).union(W).union(Y).union(X)
data.groupby("AuthorNum").count().show()
#print("Numero de Registros Dataset Limpio:",data.count(),", Atributos:",len(data.columns))

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

train, test = data.randomSplit([0.8, 0.2], seed=12345)

lr = LogisticRegression(labelCol="AuthorNum", featuresCol="features",maxIter=50, elasticNetParam=1.0, family="multinomial")

# Fit the model
lrModel = lr.fit(train)

# Print the coefficients and intercept for multinomial logistic regression
#print("Coefficients:" + str(lrModel.coefficientMatrix))

predict_train=lrModel.transform(train)
predict_test=lrModel.transform(test)

evaluator = MulticlassClassificationEvaluator(labelCol="AuthorNum", predictionCol="prediction", metricName="f1")
lr_accuracy = evaluator.evaluate(predict_test)
print("Accuracy of LogisticRegression is = %g"% (lr_accuracy))

from pyspark.ml.tuning import CrossValidator, ParamGridBuilder

paramGrid = (ParamGridBuilder()
  .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
  .addGrid(lr.maxIter, [10, 20, 50]) \
  .addGrid(lr.elasticNetParam, [0.0, 0.8]) \
  .build())

crossval = CrossValidator(estimator=lr,
                          estimatorParamMaps=paramGrid,
                          evaluator=evaluator,
                          numFolds=3)

model_lr = crossval.fit(train)


#Modelo 2
from pyspark.ml.classification import DecisionTreeClassifier
dt = DecisionTreeClassifier(labelCol="AuthorNum", featuresCol="features")
dt_model = dt.fit(train)
dt_prediction = dt_model.transform(test)

dt_accuracy = evaluator.evaluate(dt_prediction)
print("Accuracy of DecisionTreeClassifier is = %g"% (dt_accuracy))
#print("Test Error of DecisionTreeClassifier = %g " % (1.0 - dt_accuracy))
#Modelo 3

# from pyspark.ml.classification import NaiveBayes
# nb = NaiveBayes(labelCol="AuthorNum",featuresCol="features")
# nb_model = nb.fit(train)
# nb_prediction = nb_model.transform(test)
# nb_accuracy = evaluator.evaluate(nb_prediction)
# print("Accuracy of Naive bayes is = %g"%(nb_accuracy))

#Modelo 4
# from pyspark.ml.classification import GBTClassifier
# gbt = GBTClassifier(labelCol="AuthorNum", featuresCol="features",maxIter=10)
# gbt_model = gbt.fit(train)
# gbt_prediction = gbt_model.transform(test)
# #gbt_prediction.select("prediction", "Survived", "features").show()

# gbt_accuracy = evaluator.evaluate(gbt_prediction)
# print("Accuracy of Gradient-boosted tree classifie is = %g"% (gbt_accuracy))
# print("Test Error of Gradient-boosted tree classifie %g"% (1.0 - gbt_accuracy))



from pyspark.ml.classification import RandomForestClassifier
rf = DecisionTreeClassifier(labelCol="AuthorNum", featuresCol="features")
rf_model = rf.fit(train)
rf_prediction = rf_model.transform(test)

rf_accuracy = evaluator.evaluate(rf_prediction)
print("Accuracy of RandomForestClassifier is = %g"% (rf_accuracy))
#print("Test Error of RandomForestClassifier  = %g " % (1.0 - rf_accuracy))
