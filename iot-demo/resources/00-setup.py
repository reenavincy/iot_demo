# Databricks notebook source
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# DBTITLE 1,Package imports
from pyspark.sql.functions import rand, input_file_name, from_json, col
from pyspark.sql.types import *
 
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline

#ML import
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.mllib.evaluation import MulticlassMetrics
from mlflow.utils.file_utils import TempDir
import mlflow.spark
import mlflow
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from time import sleep
import re

# COMMAND ----------

# DBTITLE 1,Mount S3 bucket containing sensor data
aws_bucket_name = "iot-demo-resources"
mount_name = "iot-demo-resources"

try:
  dbutils.fs.ls("/mnt/%s" % mount_name)
except:
  print("bucket isn't mounted, mounting the demo bucket under %s" % mount_name)
  dbutils.fs.mount("s3a://%s" % aws_bucket_name, "/mnt/%s" % mount_name)


# COMMAND ----------

# DBTITLE 1,Create User-Specific database
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
print("Created variables:")
print("current_user: {}".format(current_user))
dbName = re.sub(r'\W+', '_', current_user)
path = "/Users/{}/demo".format(current_user)
dbutils.widgets.text("path", path, "path")
dbutils.widgets.text("dbName", dbName, "dbName")
print("path (default path): {}".format(path))
spark.sql("""create database if not exists {} LOCATION '{}/global_demo/tables' """.format(dbName, path))
spark.sql("""USE {}""".format(dbName))
print("dbName (using database): {}".format(dbName))

# COMMAND ----------

# DBTITLE 1,Reset tables in user's database
tables = ["turbine_bronze", "turbine_silver", "turbine_gold"]

reset_all = dbutils.widgets.get("reset_all_data") == "true" or any([not spark.catalog._jcatalog.tableExists(table) for table in ["turbine_power"]])
if reset_all:
  print("resetting data")
  for table in tables:
    spark.sql("""drop table if exists {}.{}""".format(dbName, table))

  spark.sql("""create database if not exists {} LOCATION '{}/tables' """.format(dbName, path))
  dbutils.fs.rm(path+"/turbine/bronze/", True)
  dbutils.fs.rm(path+"/turbine/silver/", True)
  dbutils.fs.rm(path+"/turbine/gold/", True)
  dbutils.fs.rm(path+"/turbine/_checkpoint", True)
      

  
else:
  print("loaded without data reset")

  
# Define the default checkpoint location to avoid managing that per stream and making it easier. In production it can be better to set the location at a stream level.
spark.conf.set("spark.sql.streaming.checkpointLocation", path+"/turbine/_checkpoint")

#Allow schema inference for auto loader
spark.conf.set("spark.databricks.cloudFiles.schemaInference.enabled", "true")

# COMMAND ----------

spark.conf.set("spark.databricks.cloudFiles.schemaInference.sampleSize.numFiles", 10)

# COMMAND ----------

# DBTITLE 1,Create "gold" tables for  ML purposes
# MAGIC %sql
# MAGIC DROP TABLE IF EXISTS turbine_gold_for_ml;
# MAGIC 
# MAGIC 
# MAGIC CREATE TABLE turbine_gold_for_ml
# MAGIC (ID double,AN3 double,AN4 double,AN5 double,AN6 double,AN7 double,AN8 double,AN9 double,AN10 double,SPEED double,TORQUE double,TIMESTAMP timestamp, STATUS string)
# MAGIC USING DELTA;
# MAGIC 
# MAGIC COPY INTO turbine_gold_for_ml FROM '/mnt/iot-demo-resources/turbine/gold-data-for-ml'
# MAGIC FILEFORMAT = PARQUET
