# Databricks notebook source
# DBTITLE 1,Widget creation
dbutils.widgets.dropdown("reset_all_data", "false", ["true", "false"], "Reset all data")

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC #Datascience/Machine Learning
# MAGIC 
# MAGIC # Wind Turbine Predictive Maintenance
# MAGIC 
# MAGIC In this notebook, we demonstrate anomaly detection for the purposes of finding damaged wind turbines. A damaged, single, inactive wind turbine costs energy utility companies thousands of dollars per day in losses.
# MAGIC 
# MAGIC 
# MAGIC Our dataset consists of vibration readings coming off sensors located in the gearboxes of wind turbines. 
# MAGIC 
# MAGIC We will use Gradient Boosted Tree Classification to predict which set of vibrations could be indicative of a failure.
# MAGIC 
# MAGIC Once the model is trained, we'll use MFLow to track its performance and save it in the registry to deploy it in production
# MAGIC 
# MAGIC 
# MAGIC 
# MAGIC *Data Source Acknowledgement: This Data Source Provided By NREL*
# MAGIC 
# MAGIC *https://www.nrel.gov/docs/fy12osti/54530.pdf*

# COMMAND ----------

# DBTITLE 1,Basic setup - Run this cell for basic setup
# MAGIC %run ./resources/00-setup $reset_all=$reset_all_data

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Preparation
# MAGIC ### Ingestion using - Auto Loader 
# MAGIC Incrementally and efficiently processes new data files as they arrive in cloud storage.
# MAGIC Provides a Structured Streaming source called cloudFiles. Given an input directory path on the cloud file storage, the cloudFiles source automatically processes new files as they arrive.
# MAGIC 
# MAGIC #### Documentation : https://docs.databricks.com/spark/latest/structured-streaming/auto-loader-json.html

# COMMAND ----------

# DBTITLE 1,Ingesting data into ML gold table using  Autoloader
goldmlDF = (spark.readStream
                .format("cloudFiles")
                .option("cloudFiles.format", "parquet")
                .option("cloudFiles.maxFilesPerTrigger", "1")  #demo only, remove in real stream
                .option("cloudFiles.schemaLocation", path+"/turbine/gold_ml")
                .option("rescuedDataColumn", "_rescued")
                .option("cloudFiles.inferColumnTypes", "true")
                .load(gold_data_for_ml_path))

goldmlDF.select("ID","AN3","AN4","AN5","AN6","AN7","AN8","AN9","AN10","SPEED","TIMESTAMP", "STATUS") \
        .writeStream.format("delta") \
        .trigger(processingTime='10 seconds') \
        .option("mergeSchema", "true").table(gold_data_for_ml_table)

# COMMAND ----------

# DBTITLE 1,List the number of files - wait until all files are loaded
dbutils.fs.ls (path+'/workshop/tables/turbine_gold_ml')

# COMMAND ----------

# MAGIC 
# MAGIC %md 
# MAGIC #### Performance Optimization
# MAGIC ###### Optimize 
# MAGIC We can improve the speed of read queries from a table by coalescing small files into larger ones. You can trigger compaction by running the OPTIMIZE command.
# MAGIC 
# MAGIC ###### Documentation : https://docs.databricks.com/delta/optimizations/file-mgmt.html
# MAGIC 
# MAGIC Note: You can also turn on autocompaction to solve small files issues on your streaming job by setting tblproperties ('delta.autoOptimize.autoCompact' = true, 'delta.autoOptimize.optimizeWrite' = true)
# MAGIC 
# MAGIC ###### Documentation : https://docs.databricks.com/spark/latest/spark-sql/language-manual/sql-ref-syntax-ddl-alter-table.html

# COMMAND ----------

# DBTITLE 1,Let us resolve the small files issue
# MAGIC %sql 
# MAGIC OPTIMIZE turbine_gold_ml

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Exploration
# MAGIC What do the distributions of sensor readings look like for our turbines? 

# COMMAND ----------

# DBTITLE 1,Display the data
dataset = spark.read.table(gold_data_for_ml_table)
display(dataset)


# COMMAND ----------

# DBTITLE 1,Analyze the data
dbutils.data.summarize(dataset)

# COMMAND ----------

# DBTITLE 1,Visualize Feature Distributions
#dropping unnecessary columns
sn.pairplot(dataset.drop("SPEED","STATUS","ID","TIMESTAMP").toPandas())

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Train Model and Track Experiments

# COMMAND ----------

#once the data is ready, we can train a model


with mlflow.start_run():
  
  training, test = dataset.limit(1000).randomSplit([0.9, 0.1], seed = 5)
  
  gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)
  grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3,4,5,10,15,25,30]).build()

  metrics = MulticlassClassificationEvaluator(metricName="f1")
  ev=BinaryClassificationEvaluator()
  cv = CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=metrics, numFolds=2)

  featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]
  stages = [VectorAssembler(inputCols=featureCols, outputCol="va"), StandardScaler(inputCol="va", outputCol="features"), StringIndexer(inputCol="STATUS", outputCol="label"), cv]
  pipeline = Pipeline(stages=stages)

  pipelineTrained = pipeline.fit(training)
  
  predictions = pipelineTrained.transform(test)
  
  metrics = MulticlassMetrics(predictions.select(['prediction', 'label']).rdd)
  metricsAUROC = ev.evaluate(predictions)
  
  
  #log your metrics (precision, recall, f1 etc) 
  #Tips: what about auto logging ?
  # mlflow.autolog() --> doesn't collect metrics on spark ML (https://www.mlflow.org/docs/latest/tracking.html#spark)
  
  mlflow.log_metric("f1",metrics.fMeasure(1.0))
  mlflow.log_metric("recall",metrics.recall(1.0))  
  mlflow.log_metric("precision",metrics.precision(1.0))
  mlflow.log_metric("AUROC",metricsAUROC)
  
  #log your model under "turbine_gbt"
  mlflow.spark.log_model(pipelineTrained, "turbine_gbt")
  mlflow.set_tag("model", "turbine_gbt")
 

# COMMAND ----------

# DBTITLE 1,Add confusion matrix to the model
  
  with TempDir() as tmp_dir:
    labels = pipelineTrained.stages[2].labels
    sn.heatmap(pd.DataFrame(metrics.confusionMatrix().toArray()), annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
    plt.suptitle("Turbine Damage Prediction. F1={:.2f}".format(metrics.fMeasure(1.0)), fontsize = 18)
    plt.xlabel("Predicted Labels")
    plt.ylabel("True Labels")
    plt.savefig(tmp_dir.path()+"/confusion_matrix.png")
    mlflow.log_artifact(tmp_dir.path()+"/confusion_matrix.png")

# COMMAND ----------

# MAGIC %md ## Save to the model registry

# COMMAND ----------

# DBTITLE 1,Getting the model having the best Accuracy from the registry
#get the best model from the registry

best_model = mlflow.search_runs(filter_string='tags.model="turbine_gbt" and attributes.status = "FINISHED"', order_by = ['metrics.AUROC DESC'], max_results=1).iloc[0]
print("Best Model run id : {}".format(best_model.run_id))

# COMMAND ----------

# DBTITLE 1, Registering the best model 

#register the model to MLFLow registry
model_registered = mlflow.register_model("runs:/"+best_model.run_id+"/turbine_gbt", "iot-model")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Flag model for Staging / Production

# COMMAND ----------

# DBTITLE 1,Transition the model version as staging/production ready
client = mlflow.tracking.MlflowClient()
print("registering model version "+model_registered.version+" as production model")

client.transition_model_version_stage(
  name=model_registered.name,
  version=model_registered.version,
  stage='Production',
)

# COMMAND ----------

# MAGIC %md #Deploying & using our model in production
# MAGIC 
# MAGIC Now that our model is in our MLFlow registry, we can start to use it in a production pipeline.

# COMMAND ----------

# MAGIC %md ### Scaling inferences using Spark 
# MAGIC We'll first see how it can be loaded as a spark UDF and called directly in a SQL function:

# COMMAND ----------

#Load the model from the registry
from pyspark.sql.functions import struct
model_udf = mlflow.pyfunc.spark_udf(spark, "models:/iot-model/Production")

#Define the model as a SQL function to be able to call it in SQL
spark.udf.register("predict", model_udf)

output_df = dataset.withColumn("prediction", model_udf(struct(*dataset.columns)))

display(output_df)

# COMMAND ----------

# MAGIC %sql
# MAGIC --Call the model in SQL using the udf registered as function
# MAGIC select *, predict(struct(AN3, AN4, AN5, AN6, AN7, AN8, AN9, AN10)) as status_forecast from turbine_gold_ml
