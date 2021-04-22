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

df = spark.read.csv('data/injection_data_v4.csv', inferSchema=True, header=True)
df = df.drop("_c0")
df.printSchema()

# major_df = df.filter(col("reason") == 0)
# minor_df = df.filter(col("reason") == 1)
# ratio = float(major_df.count()/minor_df.count())
# print("ratio: {}".format(ratio))
# sampled_majority_df = major_df.sample(False, 1/ratio)
# combined_df_2 = sampled_majority_df.unionAll(minor_df)

# Split data into train and test
df_train, df_test = df.randomSplit(weights = [0.80, 0.20], seed = 13)

# Remove nulls from our datasets
df_training = df_train.dropna()
df_testing = df_test.dropna()

"""
Building the machine learning model
"""

input_cols = ['tmpMoldZone25', 'timCool1', 'tmpBarrel2Zone3', \
       'tmpMoldZone3', 'tmpBarrel2Zone4', 'tmpFlange1', 'tmpMoldZone4', \
       'tmpBarrel2Zone1', 'tmpFlange2', 'tmpMoldZone1', 'volCushion1', \
       'tmpBarrel2Zone2', 'tmpMoldZone2', 'volCushion2', 'prsBackSpec2', \
       'prsBackSpec1', 'tmpMoldZone9', 'spdInjection2', 'tmpMoldZone7', \
       'tmpMoldZone8', 'tmpOil', 'tmpMoldZone5', 'tmpMoldZone6', \
       'tmpMoldZone19', 'tmpMoldZone18', 'volTransfer2', 'tmpMoldZone15', \
       'volTransfer1', 'tmpMoldZone14', 'tmpMoldZone17', 'tmpMoldZone16', \
       'timTransfer2', 'timTransfer1', 'velPlasticisation2', \
       'velPlasticisation1', 'timMoldClose', 'tmpBarrel1Zone5', \
       'tmpMoldZone22', 'tmpBarrel1Zone4', 'tmpMoldZone21', 'tmpMoldZone24', \
       'tmpBarrel1Zone6', 'tmpMoldZone23', 'prsPomp1', 'tmpBarrel1Zone1', \
       'prsPomp2', 'tmpBarrel1Zone3', 'tmpMoldZone20', 'tmpBarrel1Zone2', \
       'volShot1', 'volPlasticisation2', 'volShot2', 'volPlasticisation1', \
       'timFill1', 'timFill2', 'timMoldOpen', 'tmpMoldZone11', 'tmpMoldZone10', \
       'tmpMoldZone13', 'tmpMoldZone12', 'prsHoldSpec2', 'tmpNozle2', \
       'prsHoldSpec1', 'tmpNozle1', 'prsTransferSpec2', 'prsTransferSpec1', \
       'prsInjectionSpec1', 'prsInjectionSpec2', 'timCycle', 'frcClamp', \
       'timPlasticisation1', 'timPlasticisation2']

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
    .option('subscribe', 'e114_qpredict') \
    .load()

#Define schema
schema = StructType([StructField('tmpMoldZone25', DoubleType()),\
                        StructField('timCool1', DoubleType()),\
                        StructField('tmpBarrel2Zone3', DoubleType()),\
                        StructField('tmpMoldZone3', DoubleType()),\
                        StructField('tmpBarrel2Zone4', DoubleType()),\
                        StructField('tmpFlange1', DoubleType()),\
                        StructField('tmpMoldZone4', DoubleType()),\
                        StructField('tmpBarrel2Zone1', DoubleType()),\
                        StructField('tmpFlange2', DoubleType()),\
                        StructField('tmpMoldZone1', DoubleType()),\
                        StructField('volCushion1', DoubleType()),\
                        StructField('tmpBarrel2Zone2', DoubleType()),\
                        StructField('tmpMoldZone2', DoubleType()),\
                        StructField('volCushion2', DoubleType()),\
                        StructField('prsBackSpec2', DoubleType()),\
                        StructField('prsBackSpec1', DoubleType()),\
                        StructField('tmpMoldZone9', DoubleType()),\
                        StructField('spdInjection2', DoubleType()),\
                        StructField('tmpMoldZone7', DoubleType()),\
                        StructField('tmpMoldZone8', DoubleType()),\
                        StructField('tmpOil', DoubleType()),\
                        StructField('tmpMoldZone5', DoubleType()),\
                        StructField('tmpMoldZone6', DoubleType()),\
                        StructField('tmpMoldZone19', DoubleType()),\
                        StructField('tmpMoldZone18', DoubleType()),\
                        StructField('volTransfer2', DoubleType()),\
                        StructField('tmpMoldZone15', DoubleType()),\
                        StructField('volTransfer1', DoubleType()),\
                        StructField('tmpMoldZone14', DoubleType()),\
                        StructField('tmpMoldZone17', DoubleType()),\
                        StructField('tmpMoldZone16', DoubleType()),\
                        StructField('timTransfer2', DoubleType()),\
                        StructField('timTransfer1', DoubleType()),\
                        StructField('velPlasticisation2', DoubleType()),\
                        StructField('velPlasticisation1', DoubleType()),\
                        StructField('timMoldClose', DoubleType()),\
                        StructField('tmpBarrel1Zone5', DoubleType()),\
                        StructField('tmpMoldZone22', DoubleType()),\
                        StructField('tmpBarrel1Zone4', DoubleType()),\
                        StructField('tmpMoldZone21', DoubleType()),\
                        StructField('tmpMoldZone24', DoubleType()),\
                        StructField('tmpBarrel1Zone6', DoubleType()),\
                        StructField('tmpMoldZone23', DoubleType()),\
                        StructField('prsPomp1', DoubleType()),\
                        StructField('tmpBarrel1Zone1', DoubleType()),\
                        StructField('prsPomp2', DoubleType()),\
                        StructField('tmpBarrel1Zone3', DoubleType()),\
                        StructField('tmpMoldZone20', DoubleType()),\
                        StructField('tmpBarrel1Zone2', DoubleType()),\
                        StructField('volShot1', DoubleType()),\
                        StructField('volPlasticisation2', DoubleType()),\
                        StructField('volShot2', DoubleType()),\
                        StructField('volPlasticisation1', DoubleType()),\
                        StructField('timFill1', DoubleType()),\
                        StructField('timFill2', DoubleType()),\
                        StructField('timMoldOpen', DoubleType()),\
                        StructField('tmpMoldZone11', DoubleType()),\
                        StructField('tmpMoldZone10', DoubleType()),\
                        StructField('tmpMoldZone13', DoubleType()),\
                        StructField('tmpMoldZone12', DoubleType()),\
                        StructField('prsHoldSpec2', DoubleType()),\
                        StructField('tmpNozle2', DoubleType()),\
                        StructField('prsHoldSpec1', DoubleType()),\
                        StructField('tmpNozle1', DoubleType()),\
                        StructField('prsTransferSpec2', DoubleType()),\
                        StructField('prsTransferSpec1', DoubleType()),\
                        StructField('prsInjectionSpec1', DoubleType()),\
                        StructField('prsInjectionSpec2', DoubleType()),\
                        StructField('timCycle', DoubleType()),\
                        StructField('frcClamp', DoubleType()),\
                        StructField('timPlasticisation1', DoubleType()),\
                        StructField('timPlasticisation2', DoubleType())])

#Print schema to review
kafka_df.printSchema()

#Deserialize json object and apply schema
value_df = kafka_df.select(from_json(col("value").cast("string"),schema).alias("value"))

explode_df = value_df.selectExpr("value.tmpMoldZone25", "value.timCool1", \
                                "value.tmpBarrel2Zone3","value.tmpMoldZone3", "value.tmpBarrel2Zone4", \
                                "value.tmpFlange1","value.tmpMoldZone4", "value.tmpBarrel2Zone1", \
                                "value.tmpFlange2","value.tmpMoldZone1", "value.volCushion1", \
                                "value.tmpBarrel2Zone2","value.tmpMoldZone2", "value.volCushion2", \
                                "value.prsBackSpec2","value.prsBackSpec1", "value.tmpMoldZone9", \
                                "value.spdInjection2","value.tmpMoldZone7", "value.tmpMoldZone8", \
                                "value.tmpOil","value.tmpMoldZone5", "value.tmpMoldZone6", \
                                "value.tmpMoldZone19","value.tmpMoldZone18", "value.volTransfer2", \
                                "value.tmpMoldZone15","value.volTransfer1", "value.tmpMoldZone14", \
                                "value.tmpMoldZone17","value.tmpMoldZone16", "value.timTransfer2", \
                                "value.timTransfer1","value.velPlasticisation2", "value.velPlasticisation1", \
                                "value.timMoldClose","value.tmpBarrel1Zone5", "value.tmpMoldZone22", \
                                "value.tmpBarrel1Zone4","value.tmpMoldZone21", "value.tmpMoldZone24", \
                                "value.tmpBarrel1Zone6","value.tmpMoldZone23", "value.prsPomp1", \
                                "value.tmpBarrel1Zone1","value.prsPomp2", "value.tmpBarrel1Zone3", \
                                "value.tmpMoldZone20","value.tmpBarrel1Zone2", "value.volShot1", \
                                "value.volPlasticisation2","value.volShot2", "value.volPlasticisation1", \
                                "value.timFill1","value.timFill2", "value.timMoldOpen", \
                                "value.tmpMoldZone11","value.tmpMoldZone10", "value.tmpMoldZone13", \
                                "value.tmpMoldZone12","value.prsHoldSpec2", "value.tmpNozle2", \
                                "value.prsHoldSpec1","value.tmpNozle1", "value.prsTransferSpec2", \
                                "value.prsTransferSpec1","value.prsInjectionSpec1", "value.prsInjectionSpec2", \
                                "value.timCycle","value.frcClamp", "value.timPlasticisation1","value.timPlasticisation2")
#Print schema to review
#explode_df = explode_df.drop('reason')
explode_df.printSchema()
pred_results_stream = model.transform(explode_df)
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
        .option("topic", "e114_injection_pred") \
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
