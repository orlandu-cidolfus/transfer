import pyspark
from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import VectorAssembler, VectorSizeHint, StandardScaler, MinMaxScaler
from pyspark.ml.evaluation import RegressionEvaluator
import numpy as np
import json
from sklearn.metrics import classification_report, confusion_matrix


# Create spark session
spark = SparkSession\
    .builder\
    .master('local[2]')\
    .appName('injection_predictor')\
    .config('spark.jars.packages', 'org.mongodb.spark:mongo-spark-connector_2.12:2.4.1')\
    .config("spark.streaming.stopGracefullyOnShutdown", "true") \
    .getOrCreate()

df = spark.read.csv('data/117_injection_data_v1.csv', inferSchema=True, header=True)
df = df.drop("_c0")
df.printSchema()

major_df = df.filter(col("reason") == 0)
minor_df = df.filter(col("reason") == 1)
ratio = float(major_df.count()/minor_df.count())
print("ratio: {}".format(ratio))
sampled_majority_df = major_df.sample(False, 1/ratio)
combined_df_2 = sampled_majority_df.unionAll(minor_df)

# Split data into train and test
df_train, df_test = combined_df_2.randomSplit(weights = [0.80, 0.20], seed = 13)

# Remove nulls from our datasets
df_training = df_train.dropna()
df_testing = df_test.dropna()

"""
Building the machine learning model
"""

input_cols = ['timCool1', 'strCushion1', 'tmpMoldZone3', \
       'tmpFlange1', 'tmpMoldZone4', 'strPlasticisation1', 'tmpMoldZone1', \
       'tmpMoldZone2', 'spdInjection1', 'tmpMoldZone7', 'tmpMoldZone8', \
       'tmpOil', 'tmpMoldZone5', 'tmpMoldZone6', 'volTransfer1', \
       'strTransfer1', 'timTransfer1', 'timMoldClose', 'tmpBarrel1Zone5', \
       'tmpBarrel1Zone4', 'prsPomp1', 'tmpBarrel1Zone1', 'tmpBarrel1Zone3', \
       'tmpBarrel1Zone2', 'volShot1', 'timFill1', 'timMoldOpen', \
       'prsTransferHyd1', 'prsHoldSpec1', 'tmpNozle1', 'prsInjectionHyd1', \
       'timCycle', 'frcClamp', 'timPlasticisation1']

# Create feature vector
assembler = VectorAssembler(inputCols=input_cols, outputCol='assembler')
# Normalization
scaler = MinMaxScaler(inputCol="assembler",outputCol="features")
# Create the model
model_reg = RandomForestClassifier(featuresCol='features', labelCol='reason')

# Chain assembler and model into a pipleine
pipeline = Pipeline(stages=[assembler, scaler, model_reg])

# Train the Model
model = pipeline.fit(df_training)

# Make the prediction
pred_results = model.transform(df_testing)

# Evaluate model
evaluator = RegressionEvaluator(labelCol='reason', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(pred_results)

"""
Create the prediction dataset
"""
df_pred_results = pred_results['reason', 'prediction']

# Add more columns
df_pred_results = df_pred_results \
    .withColumn('RMSE', lit(rmse))
print(df_pred_results.show(5))

y_true = df_pred_results.select(['reason']).collect()
y_pred = df_pred_results.select(['prediction']).collect()

print(classification_report(y_true, y_pred))
print(confusion_matrix(y_true, y_pred))

print('INFO: Job ran successfully')
print('')

"""
Streaming Part
"""
#Read from kafka topic "injection"
kafka_df = spark \
    .readStream \
    .format('kafka') \
    .option('kafka.bootstrap.servers', "fluster-namenode.com:6667") \
    .option("startingOffsets", "earliest") \
    .option('subscribe', 'e117_qpredict') \
    .load()

#Define schema
schema = StructType([StructField('timCool1', DoubleType()),\
                        StructField('strCushion1', DoubleType()),\
                        StructField('tmpMoldZone3', DoubleType()),\
                        StructField('tmpFlange1', DoubleType()),\
                        StructField('tmpMoldZone4', DoubleType()),\
                        StructField('strPlasticisation1', DoubleType()),\
                        StructField('tmpMoldZone1', DoubleType()),\
                        StructField('tmpMoldZone2', DoubleType()),\
                        StructField('spdInjection1', DoubleType()),\
                        StructField('tmpMoldZone7', DoubleType()),\
                        StructField('tmpMoldZone8', DoubleType()),\
                        StructField('tmpOil', DoubleType()),\
                        StructField('tmpMoldZone5', DoubleType()),\
                        StructField('tmpMoldZone6', DoubleType()),\
                        StructField('volTransfer1', DoubleType()),\
                        StructField('strTransfer1', DoubleType()),\
                        StructField('timTransfer1', DoubleType()),\
                        StructField('timMoldClose', DoubleType()),\
                        StructField('tmpBarrel1Zone5', DoubleType()),\
                        StructField('tmpBarrel1Zone4', DoubleType()),\
                        StructField('prsPomp1', DoubleType()),\
                        StructField('tmpBarrel1Zone1', DoubleType()),\
                        StructField('tmpBarrel1Zone3', DoubleType()),\
                        StructField('tmpBarrel1Zone2', DoubleType()),\
                        StructField('volShot1', DoubleType()),\
                        StructField('timFill1', DoubleType()),\
                        StructField('timMoldOpen', DoubleType()),\
                        StructField('prsTransferHyd1', DoubleType()),\
                        StructField('prsHoldSpec1', DoubleType()),\
                        StructField('tmpNozle1', DoubleType()),\
                        StructField('prsInjectionHyd1', DoubleType()),\
                        StructField('timCycle', DoubleType()),\
                        StructField('frcClamp', DoubleType()),\
                        StructField('timPlasticisation1', DoubleType())])

#Print schema to review
kafka_df.printSchema()

#Deserialize json object and apply schema
value_df = kafka_df.select(from_json(col("value").cast("string"),schema).alias("value"))

explode_df = value_df.selectExpr("value.timCool1", "value.strCushion1", \
                                "value.tmpMoldZone3","value.tmpFlange1", "value.tmpMoldZone4", \
                                "value.strPlasticisation1","value.tmpMoldZone1", "value.tmpMoldZone2", \
                                "value.spdInjection1","value.tmpMoldZone7", "value.tmpMoldZone8", \
                                "value.tmpOil","value.tmpMoldZone5", "value.tmpMoldZone6", \
                                "value.volTransfer1","value.strTransfer1", "value.timTransfer1", \
                                "value.timMoldClose","value.tmpBarrel1Zone5", "value.tmpBarrel1Zone4", \
                                "value.prsPomp1","value.tmpBarrel1Zone1", "value.tmpBarrel1Zone3", \
                                "value.tmpBarrel1Zone2","value.volShot1", "value.timFill1", \
                                "value.timMoldOpen","value.prsTransferHyd1", "value.prsHoldSpec1", \
                                "value.tmpNozle1","value.prsInjectionHyd1", "value.timCycle", \
                                "value.frcClamp","value.timPlasticisation1")
#Print schema to review
#explode_df = explode_df.drop('reason')
explode_df.printSchema()
pred_results_stream = model.transform(explode_df.na.drop)
#Remove feature column
pred_results_stream_simplified = pred_results_stream.selectExpr("timCycle", "prediction")

kafka_df = pred_results_stream_simplified.select("*")
kafka_df = kafka_df.selectExpr("cast(timCycle as string) timCycle", "prediction")

kafka_target_df = kafka_df.selectExpr("timCycle as key",
                                            "to_json(struct(*)) as value")

kafka_target_df.printSchema()

nifi_query = kafka_target_df \
        .writeStream \
        .queryName("Notification Writer") \
        .format("kafka") \
        .option("kafka.bootstrap.servers", "fluster-namenode.com:6667") \
        .option("topic", "e117_injection_pred") \
        .outputMode("append") \
        .option("checkpointLocation", "chk-point-dir") \
        .start()

nifi_query.awaitTermination()
 ## Below command used to preview results on the console before inserting data to database
 #Sink result to console
#window_query = pred_results_stream_simplified.writeStream \
#      .format("console") \
#      .outputMode("append") \
#      .trigger(processingTime="10 seconds") \
#      .start()

#window_query.awaitTermination()
