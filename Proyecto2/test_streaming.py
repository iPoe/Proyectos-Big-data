from pyspark.sql import SparkSession

from pyspark.sql.types import TimestampType, StringType, StructType, StructField


spark = SparkSession.builder.master("local").appName("Avila").config("spark.some.config.option","some-value").getOrCreate()



inputPath = "/user/maria_dev/stream_files/"


schema = StructType([ StructField("time", TimestampType(), True),
                      StructField("customer", StringType(), True),
                      StructField("action", StringType(), True),
                      StructField("device", StringType(), True)])

# Create DataFrame representing data in the JSON files
inputDF = (
  spark
    .read
    .schema(schema)
    .json(inputPath)
)

inputDF.printSchema()


streamingDF = (
  spark
    .readStream
    .schema(schema)
    .option("maxFilesPerTrigger", 1)
    .json(inputPath)
)

streamingActionCountsDF = (
  streamingDF
    .groupBy(
      streamingDF.action
    )
    .count()
)

#streamingActionCountsDF.isStreaming




