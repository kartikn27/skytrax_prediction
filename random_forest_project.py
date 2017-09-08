
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
#spark = SparkSession.builder.master('local[2]').getOrCreate()
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
import pandas as pd


# In[3]:

from functools import reduce
from pyspark.sql import DataFrame

get_ipython().system('curl https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airline.csv?accessType=DOWNLOAD > airline.csv')

get_ipython().system('hdfs dfs -put airline.csv\xa0\xa0')

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
airlineCleanDF = airlineCleanDF.withColumn("label", fn.col("recommended").cast("double"))
airlineCleanDF = reduce(DataFrame.drop, ['overall_rating','seat_comfort_rating', 'cabin_staff_rating', 'food_beverages_rating', 'inflight_entertainment_rating', 'ground_service_rating', 'wifi_connectivity_rating', 'value_money_rating','recommended'], airlineCleanDF)

a1 = airlineCleanDF.where(fn.col('label')==1)  # Take data where recommendedi is 1
a0 = airlineCleanDF.where(fn.col('label')==0)  # Take data where recommendedi is 0
airlineCleanDF = a1.unionAll(a0)   # Join the two data frames

#airlineCleanDF = airlineCleanDF.withColumnRenamed('recommendedi', 'label')

airlineCleanDF.printSchema()


# In[64]:

#airlineCleanDF.show(5)
airlineCleanDF.select(fn.avg('overall_ratingf'),fn.avg('seat_comfort_ratingf'), fn.avg('cabin_staff_ratingf'), fn.avg('food_beverages_ratingf'), fn.avg('inflight_entertainment_ratingf'), fn.avg('ground_service_ratingf'), fn.avg('wifi_connectivity_ratingf'), fn.avg('value_money_ratingf')).show()
#airlineCleanDF = airlineCleanDF.fillna(0)


# In[4]:

airlineCleanDFP = airlineCleanDF.toPandas()


# In[5]:

airlineCleanDFP.overall_ratingf.fillna(6.05623311778669, inplace=True)
airlineCleanDFP.seat_comfort_ratingf.fillna(3.098869298010084, inplace=True)
airlineCleanDFP.cabin_staff_ratingf.fillna(3.32342062876573, inplace=True)
airlineCleanDFP.food_beverages_ratingf.fillna(2.8072132927233797, inplace=True)
airlineCleanDFP.inflight_entertainment_ratingf.fillna(2.392053876393633, inplace=True)
airlineCleanDFP.ground_service_ratingf.fillna(2.783941605839416, inplace=True)
airlineCleanDFP.wifi_connectivity_ratingf.fillna(2.2045855379188715, inplace=True)
airlineCleanDFP.value_money_ratingf.fillna(3.1677117572692794, inplace=True)


# In[6]:

from pyspark.sql import SQLContext
sqlCtx = SQLContext(sc)
airlineCleanDF = sqlCtx.createDataFrame(airlineCleanDFP)


# In[15]:

airlineCleanDF.where(fn.col("ground_service_ratingf").isNotNull()).count()


# In[7]:

from pyspark.ml.feature import VectorAssembler


# In[8]:

(training_df, validation_df, testing_df) = airlineCleanDF.randomSplit([0.6, 0.3, 0.1])


# In[43]:

training_df.columns[12:19]


# In[23]:

# build a pipeline for analysis
va = VectorAssembler().setInputCols(training_df.columns[12:19]).setOutputCol('features')


# In[12]:

from pyspark.ml.classification import LogisticRegression, RandomForestClassifier


# In[13]:

from pyspark.ml import Pipeline


# In[14]:

lr = LogisticRegression(regParam=0.1)


# In[24]:

lr_pipeline = Pipeline(stages=[va, lr]).fit(training_df)


# In[16]:

rf = RandomForestClassifier()


# In[25]:

rf_pipeline = Pipeline(stages=[va, rf]).fit(training_df)


# In[26]:

from pyspark.ml.evaluation import BinaryClassificationEvaluator


# In[27]:

bce = BinaryClassificationEvaluator()


# In[28]:

bce.evaluate(lr_pipeline.transform(validation_df))


# In[29]:

bce.evaluate(rf_pipeline.transform(validation_df))


# In[30]:

lr_model = lr_pipeline.stages[-1]


# In[31]:

pd.DataFrame(list(zip(airlineCleanDF.columns[12:19], lr_model.coefficients.toArray())),
            columns = ['column', 'weight']).sort_values('weight')


# In[ ]:




# In[ ]:




# In[ ]:




# In[32]:

rf_model = rf_pipeline.stages[-1]


# In[87]:

random_forest_DF = pd.DataFrame(list(zip(airlineCleanDF.columns[12:19], rf_model.featureImportances.toArray())),
            columns = ['column', 'weight']).sort_values('weight')


# In[88]:

random_forest_DF.columns


# In[83]:

vmr = random_forest_DF.loc[random_forest_DF['column'] == 'value_money_ratingf']


# In[41]:

get_ipython().magic('matplotlib inline')


# In[58]:

import matplotlib.pyplot as plt


# In[125]:

random_forest_DF


# In[129]:

random_forest_DF['features'] = ['WIFI','GROUND SERVICE','INFLGT ENTRMNT', 'SEAT COMFORT', 'FOOD' , 'STAFF' , 'VALUE FOR MONEY']


# In[130]:

random_forest_DF


# In[106]:

random_forest_DF.weight.values 


# In[104]:

random_forest_DF.index.values


# random_forest_DF.scatter(x='column', y='weight', title ='Random Forest Feature Weights' ) 

# In[133]:

random_forest_DF.plot.barh(x='features', y='weight', title ='Random Forest Feature Weights', fontsize = 7 ) 


# In[137]:

random_forest_DF.to_csv("random_forest_features.csv" , sep= ',', encoding='utf-8')


# In[138]:

get_ipython().system('ls')


# In[ ]:




# In[103]:

len(rf_model.trees)


# In[104]:

print(rf_model.trees[19].toDebugString)

