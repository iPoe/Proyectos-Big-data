#!/usr/bin/env python
# coding: utf-8

# In[4]:


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
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.sql.functions import isnan, when, count, col, lit, sum
from pyspark.sql.functions import (to_date, datediff, date_format,month)
#Se carga el conjunto de datos

spark = SparkSession.builder.master("local").appName("Avila").config("spark.some.config.option","some-value").getOrCreate()
data = spark.read.format("csv").option("header","true").option("inferSchema", "true").load(r"avila.csv")

#####################################################################################################
#PRIMER PUNTO
#DESCRIPCION DEL CONJUNTO DE DATOS INICIAL
#Imprime la cantidad de registros y atributos respectivamente
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


# In[5]:


#Se verifica la correlacion entre los atributos
#pd = data.toPandas()
#print("Correlacion entre atributos")
#print(pd.corr())

#Distribucion del atributo clasificador
print("Distribucion del atributo clasificador")
data.groupby("Author").count().show()
#####################################################################################################
#COMIENZA EL SEGUNDO PUNTO
#LIMPIEZA DE LOS DATOS

#Como se puede ver en los diagramas de cajas, el atributo F2 tiene datos que son demasiado atipicos
#Estos registros se eliminaran
print("LIMPIEZA DE LOS DATOS")
data = data.filter(data.F2<350)
print("Datos Demasiado Atipicos de F2 Eliminados:",data.count())

print("Conversion de atributos categoricos a numericos")
indexer = StringIndexer(inputCol="Author", outputCol="AuthorNum")
data = indexer.fit(data).transform(data)
data = data.drop('Author')
data.groupby("AuthorNum").count().show()

#Se balancea cada categoria (Entre 1000 y 2000 atributos cada una)
A = data.filter(data.AuthorNum == 0.0).sample(fraction=0.24)
F = data.filter(data.AuthorNum == 0.0).sample(fraction=0.53)
E = data.filter(col("AuthorNum") == 2.0).withColumn("dummy", explode(array([lit(x) for x in range(1)]))).drop('dummy')
I = data.filter(col("AuthorNum") == 3.0).withColumn("dummy", explode(array([lit(x) for x in range(1)]))).drop('dummy')
X = data.filter(col("AuthorNum") == 4.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
H = data.filter(col("AuthorNum") == 5.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
G = data.filter(col("AuthorNum") == 6.0).withColumn("dummy", explode(array([lit(x) for x in range(2)]))).drop('dummy')
D = data.filter(col("AuthorNum") == 7.0).withColumn("dummy", explode(array([lit(x) for x in range(3)]))).drop('dummy')
Y = data.filter(col("AuthorNum") == 8.0).withColumn("dummy", explode(array([lit(x) for x in range(4)]))).drop('dummy')
C = data.filter(col("AuthorNum") == 9.0).withColumn("dummy", explode(array([lit(x) for x in range(8)]))).drop('dummy')
W = data.filter(col("AuthorNum") == 10.0).withColumn("dummy", explode(array([lit(x) for x in range(17)]))).drop('dummy')
B = data.filter(col("AuthorNum") == 11.0).withColumn("dummy", explode(array([lit(x) for x in range(170)]))).drop('dummy')

#Se juntan todas las categorias balanceadas
data = A.union(B).union(C).union(D).union(E).union(F).union(G).union(H).union(I).union(W).union(Y).union(X)

print("Conjunto Balanceado")
data.groupby("AuthorNum").count().show()
print("Numero de Registros Dataset Limpio:",data.count(),", Atributos:",len(data.columns))



#####################################################################################################
#COMIENZA PUNTO 3
#Entrenamiento de modelos:
cols=data.columns
cols.remove("AuthorNum")
assembler = VectorAssembler(inputCols=cols,outputCol="features")


#Se crea el conjunto de entrenamiento y test
train, test = data.randomSplit([0.7, 0.3],seed=20)
#train=assembler.transform(train)


# In[6]:


print("ENTRENAMIENTO:")
print("Numero de Registros Train:",train.count(),", Atributos:",train.columns)
print("TEST:")
print("Numero de Registros Test:",test.count(),", Atributos:",len(test.columns))


# In[7]:


#creación baches
num_baches = 4
div = 1/num_baches
df1,df2,df3,df4 = test.randomSplit([div,div,div,div],seed=20)
df1.toPandas().to_csv('test/1.csv',index=False)
df2.toPandas().to_csv('test/2.csv',index=False) 
df3.toPandas().to_csv('test/3.csv',index=False) 
df4.toPandas().to_csv('test/4.csv',index=False)


# In[8]:


from pyspark.ml.feature import VectorAssembler,VectorSizeHint
from pyspark.ml import Pipeline
#Crear al pipeline con dos etapas
cols=data.columns
cols.remove("AuthorNum")
assembler = VectorAssembler(inputCols=cols,outputCol="features")
rf = RandomForestClassifier(labelCol="AuthorNum", featuresCol="features",numTrees=10,subsamplingRate=1,maxDepth=10)
pipeline = Pipeline(stages=[assembler,rf])
#Guardar el pipeline en la carpeta Spipeline_2
pipelineModel = pipeline.fit(train)
pipelineModel.write().overwrite().save("Spipeline_2")

