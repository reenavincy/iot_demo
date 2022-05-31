# Databricks notebook source
# DBTITLE 1,Widget creation
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# DBTITLE 1,Basic setup - Run this cell for basic setup
# MAGIC %run ./resources/00-setup $reset_all_data=$reset_all_data

# COMMAND ----------

# MAGIC %md-sandbox 
# MAGIC 
# MAGIC ## Ingesting and Cleansing the data with Delta Lake
# MAGIC To deliver final outcomes such as Wind turbine status forecasting , we need to gather, process and clean the incoming data.
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC <div style="float:left"> 
# MAGIC   
# MAGIC This can be challenging with traditional systems due to the following:
# MAGIC  * Data quality issue
# MAGIC  * Running concurrent operation
# MAGIC  * Running DELETE/UPDATE/MERGE over files
# MAGIC  * Governance (time travel, security, sharing data change, schema evolution)
# MAGIC  * Performance (lack of index, ingesting millions of small files on cloud storage)
# MAGIC  * Processing & analysing unstructured data (image, video...)
# MAGIC  * Switching between batch or streaming depending of your requirement
# MAGIC   
# MAGIC </div>
# MAGIC 
# MAGIC <img src="https://pages.databricks.com/rs/094-YMS-629/images/delta-lake-logo.png" width="200px" style="margin: 50px 0px 0px 50px"/>
# MAGIC 
# MAGIC <br style="clear: both"/>
# MAGIC 
# MAGIC ## Challenges addressed with Delta Lake
# MAGIC 
# MAGIC 
# MAGIC **Delta Lake is an open format storage layer that delivers reliability, security and performance on your data lake — for both streaming and batch operations**
# MAGIC 
# MAGIC With support for ACID transactions and schema enforcement, Delta Lake provides the reliability that traditional data lakes lack
# MAGIC 
# MAGIC * **ACID transactions** (Multiple writers can simultaneously modify a data set)
# MAGIC * **Full DML support** (UPDATE/DELETE/MERGE)
# MAGIC * **BATCH and STREAMING** support
# MAGIC * **Data quality** (Schema Enforcement, Inference and Evolution)
# MAGIC * **TIME TRAVEL** (Look back on how data looked like in the past)

# COMMAND ----------

# MAGIC %md
# MAGIC # Ingestion of IoT streaming data

# COMMAND ----------

# DBTITLE 1,Default path and database all set

#You need to run the setup cell to have these variable defined: %run ./resources/00-setup $reset_all_data=$reset_all_data
print(f"path={path}")
#Just save create the database to the current database, it's been initiliazed locally to your user to avoid conflict
print("your current database has been initialized to:")
print(sql("SELECT current_database() AS db").collect()[0]['db'])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Pull live data from Kinesis stream and ingest into Delta Lake
# MAGIC 
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2017/08/kinesis_blog_final.jpg" width="600px" style="margin: 50px 0px 0px 50px"/>
# MAGIC 
# MAGIC ###Documentation : https://docs.databricks.com/spark/latest/structured-streaming/kinesis.html

# COMMAND ----------

# DBTITLE 1,Read your stream using Kinesis connector
#Reading wind turbine's raw data from Kinesis stream and loading into a DataFrame
kinesis_readings = (spark
  .readStream
  .format("kinesis")
  .option("streamName", stream_name)
  .option("initialPosition", "earliest")
  .option("region", aws_region)
  .load()
  .withColumn('value',col("data").cast(StringType()))                
  .withColumn('key',col('partitionKey').cast(DoubleType()))         
                   )

# COMMAND ----------

# MAGIC %md
# MAGIC ##   Delta Lake Medalion Architecture
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2022/03/delta-lake-medallion-architecture-2.jpeg" width="600px" style="margin: 50px 0px 0px 50px"/>
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC 
# MAGIC ### Documentation : https://docs.databricks.com/delta/index.html

# COMMAND ----------

# MAGIC %md
# MAGIC ##1) Bronze layer: Ingest  from live Kinesis data stream into Bronze table

# COMMAND ----------

# DBTITLE 1,Write  live data from Kinesis stream into Bronze table
(kinesis_readings
 .select('key','value')
 .writeStream
 .queryName("kinesis_bronze_write") 
 .table(bronze_table)
)

# COMMAND ----------

# DBTITLE 1,Raw data is now available in Bronze Delta table
# MAGIC %sql
# MAGIC 
# MAGIC select * from turbine_bronze;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2) Silver layer: Transform JSON data into tabular data and cleanse data

# COMMAND ----------

# DBTITLE 1,Json data from Bronze Delta table is cleansed and written into Silver Delta table in tabular format
from pyspark.sql.types import StructType,StructField, StringType, DoubleType, TimestampType

jsonSchema = StructType([StructField(col, DoubleType(), False) for col in ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE", "ID"]] + [StructField("TIMESTAMP", TimestampType())])

silver_df = spark.readStream.table(bronze_table) \
     .withColumn("jsonData", from_json(col("value"), jsonSchema)) \
     .select("jsonData.*").drop('TORQUE').filter("ID IS NOT NULL") \
     .writeStream \
     .option("ignoreChanges", "true") \
     .format("delta") \
     .queryName("silver_write") \
     .trigger(processingTime='2 seconds') \
     .table(silver_table)
                

# COMMAND ----------

# DBTITLE 1,Json data is now cleansed and available in Silver Delta table in tabular format
# MAGIC %sql
# MAGIC select *
# MAGIC from turbine_silver

# COMMAND ----------

# MAGIC %md
# MAGIC ## COPY INTO - Ingest turbine status (Batch) data 
# MAGIC 
# MAGIC Loads data from a file location into a Delta table. This is a re-triable and idempotent operation—files in the source location that have already been loaded are skipped.
# MAGIC 
# MAGIC ### Documentation  : https://docs.databricks.com/spark/latest/spark-sql/language-manual/delta-copy-into.html

# COMMAND ----------

# DBTITLE 1,Displaying turbine status data
spark.read.format("parquet").load(status_data_path).display()

# COMMAND ----------

# MAGIC %sql
# MAGIC --Save the turbine status data as our turbine_status table
# MAGIC 
# MAGIC DROP TABLE IF EXISTS turbine_status;
# MAGIC 
# MAGIC CREATE TABLE turbine_status
# MAGIC (ID int, STATUS string)
# MAGIC USING DELTA;
# MAGIC 
# MAGIC COPY INTO turbine_status FROM '/mnt/iot-workshop-resources/turbine/status'
# MAGIC FILEFORMAT = PARQUET

# COMMAND ----------

# DBTITLE 1,Turbine status data is now available
# MAGIC %sql
# MAGIC SELECT * from turbine_status

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3) Gold layer: join information on Turbine status to add a label to our dataset

# COMMAND ----------

# DBTITLE 1,Join the batch turbine status (Damaged or Healthy) with streaming  Silver data
#Left join between turbine_stream and turbine_status on the 'id' key and save back the result as the "turbine_gold" table

turbine_stream = spark.readStream.table(silver_table)
turbine_status = spark.read.table(status_table)
turbine_gold = (turbine_stream.join(turbine_status, turbine_stream.ID == turbine_status.ID, 'left')).drop(turbine_status.ID)
turbine_gold.writeStream.format("delta").option("mergeSchema", "true").table(gold_table)

# COMMAND ----------

# DBTITLE 1,Turbine gold data is now available
# MAGIC %sql
# MAGIC --Our turbine gold table should be up and running! select few columns and validate
# MAGIC select ID,TIMESTAMP, AN3, SPEED, status from turbine_gold;

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Perform DELTA (DELETE/UPDATE/MERGE ) DML operations ! 

# COMMAND ----------

# MAGIC %md 
# MAGIC ## DELETE
# MAGIC 
# MAGIC 
# MAGIC Deletes the rows that match a predicate. When no predicate is provided, deletes all rows. Delete removes the data from the latest version of the Delta table but does not remove it from the physical storage
# MAGIC 
# MAGIC ## Documentation : https://docs.databricks.com/spark/latest/spark-sql/language-manual/delta-delete-from.html

# COMMAND ----------

# DBTITLE 1,Lots of things can go wrong with streaming data. Suppose we discover an issue with turbine data belonging to ID 500
# MAGIC %sql
# MAGIC Select count(*) from turbine_gold where ID = 500

# COMMAND ----------

# DBTITLE 1,DELETE - turbine data belonging to ID 500
# MAGIC %sql
# MAGIC DELETE FROM turbine_gold where ID = 500

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Delta History
# MAGIC 
# MAGIC Returns provenance information, including the operations, user, timestamp and so on, for each write to a table.
# MAGIC 
# MAGIC ## Documentation : https://docs.databricks.com/delta/delta-utility.html#delta-history

# COMMAND ----------

# MAGIC %sql 
# MAGIC DESCRIBE HISTORY turbine_gold

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Time Travel
# MAGIC 
# MAGIC Delta Lake time travel allows you to query an older snapshot of a Delta table. 
# MAGIC 
# MAGIC 
# MAGIC Time travel has many use cases including:
# MAGIC  * Re-creating analyses or reports
# MAGIC  * Re-producing outputs (for example, the output of a machine learning model)
# MAGIC  * Debugging or auditing, especially in regulated industries
# MAGIC  * Fixing mistakes in your data
# MAGIC 
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2019/02/shutterstock_1055076638-300x188.jpg" width="400px" style="margin: 50px 0px 0px 50px"/>
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC 
# MAGIC ### Documentation : https://docs.databricks.com/delta/delta-batch.html#deltatimetravel

# COMMAND ----------

# DBTITLE 1,We can travel back in time if the DELETE was accidental
# MAGIC %sql
# MAGIC select count(*) from turbine_gold where ID=500

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from turbine_gold  VERSION AS OF 1 where ID=500

# COMMAND ----------

# MAGIC 
# MAGIC %md 
# MAGIC ## MERGE
# MAGIC 
# MAGIC Merges a set of updates, insertions, and deletions based on a source table into a target Delta table
# MAGIC 
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/09/blog-diving-delta-4.png" width="400px" style="margin: 50px 0px 0px 50px"/>
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC 
# MAGIC ### Documentation : https://docs.databricks.com/spark/latest/spark-sql/language-manual/delta-merge-into.html

# COMMAND ----------

# DBTITLE 1,We can restore previous version
# MAGIC %sql
# MAGIC MERGE INTO turbine_gold
# MAGIC USING turbine_gold VERSION AS OF 1 AS restore_version
# MAGIC ON turbine_gold.id=restore_version.id AND turbine_gold.timestamp=restore_version.timestamp
# MAGIC WHEN NOT MATCHED THEN INSERT *

# COMMAND ----------

# MAGIC %sql
# MAGIC select count(*) from turbine_gold where ID=500

# COMMAND ----------

# MAGIC %md 
# MAGIC ## UPDATE
# MAGIC 
# MAGIC Updates the column values for the rows that match a predicate. When no predicate is provided, update the column values for all rows
# MAGIC 
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC <img src="https://databricks.com/wp-content/uploads/2020/09/blog-diving-delta-3.gif" width="400px" style="margin: 50px 0px 0px 50px"/>
# MAGIC 
# MAGIC <br/>
# MAGIC 
# MAGIC 
# MAGIC ### Documentation : https://docs.databricks.com/spark/latest/spark-sql/language-manual/delta-update.html

# COMMAND ----------

# DBTITLE 1,Before Update value
# MAGIC %sql
# MAGIC 
# MAGIC SELECT ID,STATUS FROM turbine_gold where ID=300;

# COMMAND ----------

# DBTITLE 1,UPDATE - turbine status value
# MAGIC %sql
# MAGIC 
# MAGIC UPDATE turbine_gold SET STATUS ='damaged' where ID=300

# COMMAND ----------

# DBTITLE 1,After Update value
# MAGIC %sql
# MAGIC 
# MAGIC SELECT ID,STATUS FROM turbine_gold where ID=300;

# COMMAND ----------

# MAGIC %sql 
# MAGIC DESCRIBE HISTORY turbine_gold

# COMMAND ----------

# DBTITLE 1,Perform Time Travel  before and after UPDATE
# MAGIC %sql
# MAGIC SELECT ID,STATUS from turbine_gold VERSION AS OF 4 where ID=300

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT ID,STATUS from turbine_gold where ID=300

# COMMAND ----------

# MAGIC %sql 
# MAGIC select * from turbine_gold

# COMMAND ----------

# MAGIC %md
# MAGIC ### Don't forget to Cancel all the streams once your workshop is over

# COMMAND ----------

for s in spark.streams.active:
  s.stop()
