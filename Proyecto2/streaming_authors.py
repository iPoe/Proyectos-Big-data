#!/usr/bin/env python
# coding: utf-8

# In[1]:


from pyspark.ml import PipelineModel

StreamPipeline = PipelineModel.load("Spipeline")


# In[2]:


from pyspark.sql import SparkSession
from pyspark import SparkContext
from pyspark.sql.types import TimestampType, StringType, StructType, StructField
from pyspark.ml import Pipeline
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.streaming import StreamingContext
from operator import attrgetter

from pyspark.ml.evaluation import MulticlassClassificationEvaluator

from pyspark.streaming import StreamingContext




spark = SparkSession.builder.master("local").appName("Avila-stream").config("spark.some.config.option","some-value").getOrCreate()
schema = "F1 DOUBLE, F2 DOUBLE, F3 DOUBLE, F4 DOUBLE, F5 DOUBLE, F6 DOUBLE, F7 DOUBLE, F8 DOUBLE, F9 DOUBLE, F10 DOUBLE, AuthorNum DOUBLE"


streamingDF = (
  spark
    .readStream
    .schema(schema)
    .option("header","true")
    .option("maxFilesPerTrigger", 1)
    .csv("test/")
)

evaluator = MulticlassClassificationEvaluator(labelCol="AuthorNum",predictionCol="prediction", metricName="accuracy")


def train_df(df,epoch_id):
    #df.show(5)
    print("----WORKING ON BATCH----")
    print(".................")
    print("# REGISTROS EN BATCH:",df.count(),", Atributos:",len(df.columns))
    prediction = StreamPipeline.transform(df)
    dt_accuracy = evaluator.evaluate(prediction)
    print("----TEST STREAMING RESULTS----")
    print("----BATCH PREDICTIONS---")
    print("Accuracy of RandomForest is = {}" .format(dt_accuracy))

query = streamingDF.writeStream.foreachBatch(train_df).start()   
query.awaitTermination()
    
    


