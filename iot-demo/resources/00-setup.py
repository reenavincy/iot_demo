# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# DBTITLE 1,Package imports
import re
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import *

#ML import
import seaborn as sn
import mlflow
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.feature import VectorAssembler, StandardScaler, StringIndexer
from pyspark.mllib.evaluation import MulticlassMetrics
from mlflow.utils.file_utils import TempDir
import pandas as pd
import matplotlib.pyplot as plt

# COMMAND ----------

# DBTITLE 1,Mount S3 bucket containing sensor data
# Define the values for the below variables, ensure the aws_bucket_name variable's value matches with your s3 bucket name in your AWS account
aws_bucket_name = "<SPECIFY_YOUR_AWS_ACCOUNT_ID>-iot-workshop-resources"
mount_name = "iot-workshop-resources"

try:
  dbutils.fs.ls("/mnt/%s" % mount_name)
except:
  print("bucket isn't mounted, mounting the demo bucket under %s" % mount_name)
  dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % mount_name)


# COMMAND ----------

# DBTITLE 1,Create table specific variables 
bronze_table = 'turbine_bronze'
silver_table = 'turbine_silver'
gold_table = 'turbine_gold'
status_data_path = '/mnt/iot-workshop-resources/turbine/status'
status_table = 'turbine_status'
gold_data_for_ml_path = '/mnt/iot-workshop-resources/turbine/gold-data-for-ml'
gold_data_for_ml_table = 'turbine_gold_ml'

# COMMAND ----------

# DBTITLE 1,Create User-Specific database
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
print("Created User-Specific variables:")
print("current_user: {}".format(current_user))
dbName = re.sub(r'\W+', '_', current_user)
path = "/Users/{}/iot".format(current_user)
dbutils.widgets.text("path", path, "path")
dbutils.widgets.text("dbName", dbName, "dbName")
print("path (default path): {}".format(path))
spark.sql("""create database if not exists {} LOCATION '{}/workshop/tables' """.format(dbName, path))
spark.sql("""USE {}""".format(dbName))
print("dbName (using database): {}".format(dbName))

# COMMAND ----------

# DBTITLE 1,Create Kinesis Stream Variables
# Define the values for the below variables, aws_region variable's value  and stream_name variable's value matches with your AWS region and Kinesis stream name in your AWS account
aws_region = 'eu-west-1'
stream_name = 'simulated-iot-data-stream'
print("Kinesis Stream variables:")
print("AWS region: {}".format(aws_region))
print("Kinesis Stream name: {}".format(stream_name))

# COMMAND ----------

# DBTITLE 1,Reset tables in user's database
tables = ["turbine_bronze", "turbine_silver", "turbine_gold", "turbine_gold_ml"]

reset_all = dbutils.widgets.get("reset_all_data") == "true"
if reset_all:
  print("Received value for resetting all data as: {}".format(reset_all))
  print("resetting data")
  for table in tables:
    spark.sql("""drop table if exists {}.{}""".format(dbName, table))

  spark.sql("""create database if not exists {} LOCATION '{}/workshop/tables' """.format(dbName, path))
  dbutils.fs.rm(path+"/turbine/bronze/", True)
  dbutils.fs.rm(path+"/turbine/silver/", True)
  dbutils.fs.rm(path+"/turbine/gold/", True)
  dbutils.fs.rm(path+"/turbine/gold_ml/", True)
  dbutils.fs.rm(path+"/turbine/_checkpoint", True)
      

  
else:
  print("Received value for resetting all data as: {}".format(reset_all))
  print("loaded without data reset")

  
# Define the default checkpoint location to avoid managing that per stream and making it easier. In production it can be better to set the location at a stream level.
spark.conf.set("spark.sql.streaming.checkpointLocation", path+"/turbine/_checkpoint")

#Allow schema inference for auto loader
spark.conf.set("spark.databricks.cloudFiles.schemaInference.enabled", "true")
spark.conf.set("spark.databricks.cloudFiles.schemaInference.sampleSize.numFiles", 10)
