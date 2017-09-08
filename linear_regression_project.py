
# coding: utf-8

# In[1]:

# Run this cell below only once:
import findspark
findspark.init()

# import the library
import pyspark
from pyspark.sql import SparkSession
# for Spark 2.0, we have a unified entry point to the cluster
spark = SparkSession.builder.    getOrCreate()
# for previous versions, we can simulate SparkContext and SQLContext
sc = spark.sparkContext
sqlContext = spark

# Display information about current execution
spark.conf.get('spark.org.apache.hadoop.yarn.server.webproxy.amfilter.AmIpFilter.param.PROXY_URI_BASES')


# In[2]:

from pyspark.sql import functions as fn
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml import feature
from pyspark.ml import regression
from functools import reduce
from pyspark.sql import DataFrame


# In[3]:

import pandas as pd
import io
import requests


# In[8]:

get_ipython().system('curl https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airline.csv?accessType=DOWNLOAD > airline.csv')

get_ipython().system('hdfs dfs -put airline.csv\xa0\xa0')


# In[4]:

airlineDF = spark.read.csv('airline.csv')
airlineNewDF = airlineDF.select(fn.col("_c0").alias("airline_name"), fn.col("_c1").alias("link"), fn.col("_c2").alias("title"), fn.col("_c3").alias("author"), fn.col("_c4").alias("author_country"), fn.col("_c5").alias("date"), fn.col("_c6").alias("content"), fn.col("_c7").alias("aircraft"), fn.col("_c8").alias("type_traveller"), fn.col("_c9").alias("cabin_flown"), fn.col("_c10").alias("route"), fn.col("_c11").alias("overall_rating"), fn.col("_c12").alias("seat_comfort_rating"), fn.col("_c13").alias("cabin_staff_rating"), fn.col("_c14").alias("food_beverages_rating"), fn.col("_c15").alias("inflight_entertainment_rating"), fn.col("_c16").alias("ground_service_rating"), fn.col("_c17").alias("wifi_connectivity_rating"), fn.col("_c18").alias("value_money_rating"), fn.col("_c19").alias("recommended"))

airlineCleanDF = airlineNewDF.withColumn("overall_ratingf", fn.col("overall_rating").cast("float"))
airlineCleanDF = airlineCleanDF.withColumn("seat_comfort_ratingf", fn.col("seat_comfort_rating").cast("float"))
airlineCleanDF = airlineCleanDF.withColumn("cabin_staff_ratingf", fn.col("cabin_staff_rating").cast("float"))
airlineCleanDF = airlineCleanDF.withColumn("food_beverages_ratingf", fn.col("food_beverages_rating").cast("float"))
airlineCleanDF = airlineCleanDF.withColumn("inflight_entertainment_ratingf", fn.col("inflight_entertainment_rating").cast("float"))
airlineCleanDF = airlineCleanDF.withColumn("ground_service_ratingf", fn.col("ground_service_rating").cast("float"))
airlineCleanDF = airlineCleanDF.withColumn("wifi_connectivity_ratingf", fn.col("wifi_connectivity_rating").cast("float"))
airlineCleanDF = airlineCleanDF.withColumn("value_money_ratingf", fn.col("value_money_rating").cast("float"))
airlineCleanDF = airlineCleanDF.withColumn("recommendedi", fn.col("recommended").cast("integer"))

airlineCleanDF = reduce(DataFrame.drop, ['overall_rating','seat_comfort_rating', 'cabin_staff_rating', 'food_beverages_rating', 'inflight_entertainment_rating', 'ground_service_rating', 'wifi_connectivity_rating', 'value_money_rating','recommended'], airlineCleanDF)

# Take data where recommendedi is 1
a1 = airlineCleanDF.where(fn.col('recommendedi')==1)
# Take data where recommendedi is 0
a0 = airlineCleanDF.where(fn.col('recommendedi')==0)

# Join the two data frames
airlineCleanDF = a1.unionAll(a0)


# In[6]:

airlineCleanDFP = airlineCleanDF.toPandas()


# In[8]:

airlineCleanDFP.overall_ratingf.fillna(6.05623311778669, inplace=True)
airlineCleanDFP.seat_comfort_ratingf.fillna(3.098869298010084, inplace=True)
airlineCleanDFP.cabin_staff_ratingf.fillna(3.32342062876573, inplace=True)
airlineCleanDFP.food_beverages_ratingf.fillna(2.8072132927233797, inplace=True)
airlineCleanDFP.inflight_entertainment_ratingf.fillna(2.392053876393633, inplace=True)
airlineCleanDFP.ground_service_ratingf.fillna(2.783941605839416, inplace=True)
airlineCleanDFP.wifi_connectivity_ratingf.fillna(2.2045855379188715, inplace=True)
airlineCleanDFP.value_money_ratingf.fillna(3.1677117572692794, inplace=True)


# In[9]:

from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
airlineCleanDF = sqlCtx.createDataFrame(airlineCleanDFP)


# In[10]:

training, validation, testing = airlineCleanDF.randomSplit([0.6, 0.3, 0.1], seed=0)


# In[9]:

# 2. Linear regression with avg_overall
vaAvgOverall = feature.VectorAssembler(inputCols=['overall_ratingf', 'seat_comfort_ratingf', 'cabin_staff_ratingf', 'food_beverages_ratingf', 'inflight_entertainment_ratingf', 'ground_service_ratingf', 'wifi_connectivity_ratingf', 'value_money_ratingf'], outputCol='features')
lrAvgOverall = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipelineAvgOverall = Pipeline(stages=[vaAvgOverall, lrAvgOverall])
pipeline_modelAvgOverall = pipelineAvgOverall.fit(training)
pipeline_modelAvgOverall.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE_Avg_Overall')).    show()


# In[12]:

# 2. Linear regression with avg_overall
vaAvgOverall = feature.VectorAssembler(inputCols=['seat_comfort_ratingf', 'cabin_staff_ratingf', 'food_beverages_ratingf', 'inflight_entertainment_ratingf', 'value_money_ratingf'], outputCol='features')
lrAvgOverall = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipelineAvgOverall = Pipeline(stages=[vaAvgOverall, lrAvgOverall])
pipeline_modelAvgOverall = pipelineAvgOverall.fit(training)
pipeline_modelAvgOverall.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE_Avg_Overall')).    show()


# In[17]:

# 2. Linear regression with avg_overall
vaAvgOverall = feature.VectorAssembler(inputCols=['seat_comfort_ratingf', 'cabin_staff_ratingf', 'food_beverages_ratingf', 'inflight_entertainment_ratingf', 'ground_service_ratingf', 'wifi_connectivity_ratingf', 'value_money_ratingf'], outputCol='features')
lrAvgOverall = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipelineAvgOverall = Pipeline(stages=[vaAvgOverall, lrAvgOverall])
pipeline_modelAvgOverall = pipelineAvgOverall.fit(training)
pipeline_modelAvgOverall.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE_Avg_Overall')).    show()


# In[18]:

# 2. Linear regression with avg_overall
vaAvgOverall = feature.VectorAssembler(inputCols=['cabin_staff_ratingf', 'food_beverages_ratingf', 'inflight_entertainment_ratingf', 'ground_service_ratingf', 'wifi_connectivity_ratingf', 'value_money_ratingf'], outputCol='features')
lrAvgOverall = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipelineAvgOverall = Pipeline(stages=[vaAvgOverall, lrAvgOverall])
pipeline_modelAvgOverall = pipelineAvgOverall.fit(validation)
pipeline_modelAvgOverall.transform(training).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE_Avg_Overall')).    show()


# In[10]:

# 2. Linear regression with avg_overall
vaAvgOverall = feature.VectorAssembler(inputCols=['food_beverages_ratingf', 'inflight_entertainment_ratingf', 'value_money_ratingf'], outputCol='features')
lrAvgOverall = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipelineAvgOverall = Pipeline(stages=[vaAvgOverall, lrAvgOverall])
pipeline_modelAvgOverall = pipelineAvgOverall.fit(training)
pipeline_modelAvgOverall.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE_Avg_Overall')).    show()


# In[12]:

# 2. Linear regression with avg_overall
vaAvgOverall = feature.VectorAssembler(inputCols=['overall_ratingf','value_money_ratingf'], outputCol='features')
lrAvgOverall = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipelineAvgOverall = Pipeline(stages=[vaAvgOverall, lrAvgOverall])
pipeline_modelAvgOverall = pipelineAvgOverall.fit(training)
pipeline_modelAvgOverall.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE_Avg_Overall')).    show()


# In[ ]:

# 2. Linear regression with avg_overall
vaAvgOverall = feature.VectorAssembler(inputCols=['overall_ratingf', 'seat_comfort_ratingf', 'cabin_staff_ratingf', 'food_beverages_ratingf', 'inflight_entertainment_ratingf', 'ground_service_ratingf', 'wifi_connectivity_ratingf', 'value_money_ratingf'], outputCol='features')
lrAvgOverall = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipelineAvgOverall = Pipeline(stages=[vaAvgOverall, lrAvgOverall])
pipeline_modelAvgOverall = pipelineAvgOverall.fit(training)
pipeline_modelAvgOverall.transform(training).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE_Avg_Overall')).    show()


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[12]:

va = feature.VectorAssembler(inputCols=['overall_ratingf', 'seat_comfort_ratingf', 'cabin_staff_ratingf', 'food_beverages_ratingf', 'inflight_entertainment_ratingf', 'ground_service_ratingf', 'wifi_connectivity_ratingf', 'value_money_ratingf'], outputCol='features')



# In[14]:

# scalar = feature.StandardScaler(withMean=True, withStd=True, inputCol='features', outputCol='scalarFeatures')


# In[20]:

lr = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')


# In[21]:

pipeline = Pipeline(stages=[va, lr])


# In[22]:

pipeline_model = pipeline.fit(training)


# In[23]:

pipeline_model.stages[1].coefficients


# In[24]:

pipeline_model.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE_all_vars')).    show()


# In[25]:

va1 = feature.VectorAssembler(inputCols=['overall_ratingf', 'cabin_staff_ratingf', 'food_beverages_ratingf', 'inflight_entertainment_ratingf', 'ground_service_ratingf', 'wifi_connectivity_ratingf', 'value_money_ratingf'], outputCol='features')
lr1 = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipeline1 = Pipeline(stages=[va1, lr1])
pipeline_model1 = pipeline1.fit(training)
pipeline_model1.stages[1].coefficients


# In[26]:

pipeline_model1.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE')).    show()


# In[27]:

va2 = feature.VectorAssembler(inputCols=['overall_ratingf', 'cabin_staff_ratingf', 'food_beverages_ratingf', 'ground_service_ratingf', 'wifi_connectivity_ratingf', 'value_money_ratingf'], outputCol='features')
lr2 = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipeline2 = Pipeline(stages=[va2, lr2])
pipeline_model2 = pipeline2.fit(training)
pipeline_model2.stages[1].coefficients


# In[28]:

pipeline_model2.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE')).    show()


# In[13]:

va3 = feature.VectorAssembler(inputCols=['overall_ratingf', 'cabin_staff_ratingf', 'ground_service_ratingf', 'wifi_connectivity_ratingf', 'value_money_ratingf'], outputCol='features')
lr3 = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipeline3 = Pipeline(stages=[va3, lr3])
pipeline_model3 = pipeline3.fit(training)
pipeline_model3.stages[1].coefficients


# In[30]:

pipeline_model3.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE')).    show()


# In[32]:

va4 = feature.VectorAssembler(inputCols=['overall_ratingf', 'cabin_staff_ratingf', 'food_beverages_ratingf', 'wifi_connectivity_ratingf', 'value_money_ratingf'], outputCol='features')
lr4 = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipeline4 = Pipeline(stages=[va4, lr4])
pipeline_model4 = pipeline4.fit(training)
pipeline_model4.stages[1].coefficients


# In[33]:

pipeline_model4.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE')).    show()


# In[34]:

va5 = feature.VectorAssembler(inputCols=['overall_ratingf', 'cabin_staff_ratingf', 'wifi_connectivity_ratingf', 'value_money_ratingf'], outputCol='features')
lr5 = regression.LinearRegression(featuresCol='features', labelCol='recommendedi')
pipeline5 = Pipeline(stages=[va5, lr5])
pipeline_model5 = pipeline5.fit(training)
pipeline_model5.stages[1].coefficients


# In[35]:

pipeline_model5.transform(validation).    select(fn.avg((fn.col('prediction') - fn.col('recommendedi'))**2).alias('MSE')).    show()


# In[36]:

pipeline_model3.transform(testing).select('airline_name', 'prediction', 'recommendedi').where(fn.col('recommendedi')==0).show(20)

