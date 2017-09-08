
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

get_ipython().system('curl https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airport.csv?accessType=DOWNLOAD > airport.csv')

get_ipython().system('hdfs dfs -put airport.csv   ')


# In[3]:

from functools import reduce
from pyspark.sql import DataFrame
airportDF = spark.read.csv('airport.csv')

airportNewDF = airportDF.select(fn.col("_c0").alias("airport_name"), fn.col("_c1").alias("link"), fn.col("_c2").alias("title"), fn.col("_c3").alias("author"), fn.col("_c4").alias("author_country"), fn.col("_c5").alias("date"), fn.col("_c6").alias("content"), fn.col("_c7").alias("experience_airport"), fn.col("_c8").alias("date_visit"), fn.col("_c9").alias("type_traveller"), fn.col("_c10").alias("overall_rating"), fn.col("_c11").alias("queuing_rating"), fn.col("_c12").alias("terminal_cleanliness_rating"), fn.col("_c13").alias("terminal_seating_rating"), fn.col("_c14").alias("terminal_signs_rating"), fn.col("_c15").alias("food_beverages_rating"), fn.col("_c16").alias("airport_shopping_rating"), fn.col("_c17").alias("wifi_connectivity_rating"), fn.col("_c18").alias("airport_staff_rating"), fn.col("_c19").alias("recommended"))

airportCleanDF = airportNewDF.withColumn("overall_ratingf", fn.col("overall_rating").cast("float"))
airportCleanDF = airportCleanDF.withColumn("queuing_ratingf", fn.col("queuing_rating").cast("float"))
airportCleanDF = airportCleanDF.withColumn("terminal_cleanliness_ratingf", fn.col("terminal_cleanliness_rating").cast("float"))
airportCleanDF = airportCleanDF.withColumn("terminal_seating_ratingf", fn.col("terminal_seating_rating").cast("float"))
airportCleanDF = airportCleanDF.withColumn("terminal_signs_ratingf", fn.col("terminal_signs_rating").cast("float"))
airportCleanDF = airportCleanDF.withColumn("food_beverages_ratingf", fn.col("food_beverages_rating").cast("float"))
airportCleanDF = airportCleanDF.withColumn("airport_shopping_ratingf", fn.col("airport_shopping_rating").cast("float"))
airportCleanDF = airportCleanDF.withColumn("wifi_connectivity_ratingf", fn.col("wifi_connectivity_rating").cast("float"))
airportCleanDF = airportCleanDF.withColumn("airport_staff_ratingf", fn.col("airport_staff_rating").cast("float"))
airportCleanDF = airportCleanDF.withColumn("recommendedi", fn.col("recommended").cast("integer"))

a1 = airportCleanDF.where(fn.col('recommendedi')==1)  # Take data where recommendedi is 1
a0 = airportCleanDF.where(fn.col('recommendedi')==0)  # Take data where recommendedi is 0
airportCleanDF = a1.unionAll(a0)   # Join the two data frames

airportCleanDF = reduce(DataFrame.drop, ['overall_rating','queuing_rating', 'terminal_cleanliness_rating', 'terminal_seating_rating', 'terminal_signs_rating', 'food_beverages_rating', 'airport_shopping_rating', 'wifi_connectivity_rating', 'airport_staff_rating','recommended'], airportCleanDF)


# In[93]:

airportCleanDF.count()


# In[4]:

airportCleanDF = airportCleanDF.na.drop(subset=["content"])
#airportCleanDF = airportCleanDF.na.drop(subset=["recommendedi"])


# In[5]:

from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer().setInputCol('content').setOutputCol('words')


# In[6]:

#airportCleanDF = airportCleanDF.na.drop(subset=["content"]) # Remove rows with NULL in column 'content'
tokenizer.transform(airportCleanDF)


# In[7]:

from pyspark.ml.feature import CountVectorizer


# In[8]:

count_vectorizer_estimator = CountVectorizer().setInputCol('words').setOutputCol('features')
count_vectorizer_transformer = count_vectorizer_estimator.fit(tokenizer.transform(airportCleanDF))


# In[9]:

from pyspark.ml import Pipeline


# In[10]:

pipeline_cv_estimator = Pipeline(stages=[tokenizer, count_vectorizer_estimator])
pipeline_cv_transformer = pipeline_cv_estimator.fit(airportCleanDF)
#pipeline_cv_transformer.transform(airlineCleanDF).show()


# In[11]:

sentiments_df = spark.read.parquet('/datasets/sentiment_analysis/sentiments.parquet')


# In[12]:

sentiments_df.printSchema()


# In[13]:

from pyspark.ml.feature import RegexTokenizer
tokenizer = RegexTokenizer().setGaps(False)  .setPattern("\\p{L}+")  .setInputCol("content")  .setOutputCol("words")


# In[14]:

content_words_df = tokenizer.transform(airportCleanDF)


# In[15]:

content_words_df.select('airport_name', fn.explode('words').alias('word')).show(5)


# In[16]:

content_word_sentiment_df = content_words_df.    select('airport_name', fn.explode('words').alias('word')).    join(sentiments_df, 'word')


# In[17]:

simple_sentiment_prediction_df = content_word_sentiment_df.    groupBy('airport_name').    agg(fn.avg('sentiment').alias('avg_sentiment')).    withColumn('predicted', fn.when(fn.col('avg_sentiment') > 0, 1.0).otherwise(0.))


# In[18]:

simple_sentiment_prediction_df.show(5)


# In[19]:

airportCleanDF.    join(simple_sentiment_prediction_df, 'airport_name').    select(fn.expr('float(recommendedi = predicted)').alias('correct')).    select(fn.avg('correct')).    show()


# In[20]:

#### DATA DRIVEN APPROACH
#### DATA DRIVEN APPROACH
#### DATA DRIVEN APPROACH


# In[21]:

# we obtain the stop words from a website
import requests
stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()
len(stop_words)


# In[22]:

from pyspark.ml.feature import StopWordsRemover
sw_filter = StopWordsRemover()  .setStopWords(stop_words)  .setCaseSensitive(False)  .setInputCol("words")  .setOutputCol("filtered")


# In[23]:

from pyspark.ml.feature import CountVectorizer

# we will remove words that appear in 5 docs or less
cv = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17)  .setInputCol("filtered")  .setOutputCol("tf")


# In[24]:

# we now create a pipelined transformer
cv_pipeline = Pipeline(stages=[tokenizer, sw_filter, cv]).fit(airportCleanDF)


# In[25]:

from pyspark.ml.feature import IDF
idf = IDF().    setInputCol('tf').    setOutputCol('tfidf')


# In[26]:

idf_pipeline = Pipeline(stages=[cv_pipeline, idf]).fit(airportCleanDF)


# In[27]:

tfidf_df = idf_pipeline.transform(airportCleanDF)


# In[28]:

tfidf_df.printSchema()


# In[29]:

### DATA SCIENCE PIPELINE FOR ESTIMATING SENITMENTS
### DATA SCIENCE PIPELINE FOR ESTIMATING SENITMENTS
### DATA SCIENCE PIPELINE FOR ESTIMATING SENITMENTS


# In[30]:

training_df, validation_df, testing_df = airportCleanDF.randomSplit([0.6, 0.3, 0.1], seed=0)


# In[31]:

[training_df.count(), validation_df.count(), testing_df.count()]


# In[32]:

#  Logistic Regression
from pyspark.ml.classification import LogisticRegression


# In[33]:

lr = LogisticRegression().    setLabelCol('recommendedi').    setFeaturesCol('tfidf').    setRegParam(0.0).    setMaxIter(100).    setElasticNetParam(0.)


# In[34]:

training_df.printSchema()


# In[35]:

lr_pipeline = Pipeline(stages=[idf_pipeline, lr]).fit(training_df)


# In[36]:

lr_pipeline.transform(validation_df).    select(fn.expr('float(prediction = recommendedi)').alias('correct')).    select(fn.avg('correct')).show()


# In[37]:

import pandas as pd
vocabulary = idf_pipeline.stages[0].stages[-1].vocabulary
weights = lr_pipeline.stages[-1].coefficients.toArray()
coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights})


# In[38]:

coeffs_df.sort_values('weight').head(5)


# In[39]:

coeffs_df.sort_values('weight', ascending=False).head(5)


# In[40]:

lambda_par = 0.02
alpha_par = 0.3
en_lr = LogisticRegression().        setLabelCol('recommendedi').        setFeaturesCol('tfidf').        setRegParam(lambda_par).        setMaxIter(100).        setElasticNetParam(alpha_par)


# In[41]:

en_lr_estimator = Pipeline(
    stages=[tokenizer, sw_filter, cv, idf, en_lr])


# In[42]:

en_lr_pipeline = en_lr_estimator.fit(training_df)


# In[43]:

en_lr_pipeline.transform(validation_df).select(fn.avg(fn.expr('float(prediction = recommendedi)'))).show()


# In[44]:

en_weights = en_lr_pipeline.stages[-1].coefficients.toArray()
en_coeffs_df = pd.DataFrame({'word': en_lr_pipeline.stages[2].vocabulary, 'weight': en_weights})


# In[45]:

en_coeffs_df.sort_values('weight').head(15)


# In[46]:

en_coeffs_df.sort_values('weight', ascending=False).head(15)


# In[47]:

en_coeffs_df.query('weight == 0.0').shape


# In[48]:

en_coeffs_df.query('weight == 0.0').shape[0]/en_coeffs_df.shape[0]


# In[49]:

en_coeffs_df.query('weight == 0.0').head(15)


# In[50]:

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


# In[51]:

en_lr_estimator.getStages()


# In[52]:

grid = ParamGridBuilder().    addGrid(en_lr.regParam, [0., 0.01, 0.02]).    addGrid(en_lr.elasticNetParam, [0., 0.2, 0.4]).    build()


# In[53]:

grid


# In[54]:

all_models = []
for j in range(len(grid)):
    print("Fitting model {}".format(j+1))
    model = en_lr_estimator.fit(training_df, grid[j])
    all_models.append(model)


# In[55]:

# estimate the accuracy of each of them:
accuracies = [m.    transform(validation_df).    select(fn.avg(fn.expr('float(recommendedi = prediction)')).alias('accuracy')).    first().    accuracy for m in all_models]


# In[56]:

accuracies


# In[57]:

import numpy as np


# In[58]:

best_model_idx = np.argmax(accuracies)


# In[59]:

grid[best_model_idx]


# In[60]:

best_model = all_models[best_model_idx]


# In[61]:

accuracies[best_model_idx]


# In[62]:

bestModelDF = best_model.transform(airportCleanDF)


# In[63]:

bestModelDF.printSchema()


# In[64]:

bestModelDF.select('airport_name', 'content', 'prediction').show(30)


# In[65]:

# From best fit model select only the airlines where review count is greater than 500

bestModelDF.groupBy('airport_name').    agg(fn.count('airport_name').alias('airport_review_count')).    where(fn.col('airport_review_count')>200).    orderBy(fn.desc('airport_review_count'))


# In[165]:

t = bestModelDF.groupBy('airport_name').    agg(fn.count('airport_name').alias('airport_review_count')).    where(fn.col('airport_review_count')>200)


# In[66]:

get_ipython().magic('matplotlib inline')


# In[67]:

import seaborn


# In[69]:

sentiment_pd = best_model.    transform(airportCleanDF).    groupby('airport_name').    agg(fn.avg('prediction').alias('prediction'), 
        (2*fn.stddev('prediction')/fn.sqrt(fn.count('*'))).alias('err')).\
    toPandas()


# In[ ]:




# In[ ]:




# In[70]:

arr = ["london-heathrow-airport", "london-stansted-airport", "manchester-airport", "paris-cdg-airport", "dubai-airport", "luton-airport", "london-gatwick-airport", "bangkok-suvarnabhumi-airport","frankfurt-main-airport"]


# In[171]:

bestModelPD500 = sentiment_pd.loc[sentiment_pd['airport_name'].isin(arr)]


# In[172]:

bestModelPD500


# In[177]:

bestModelPD500.plot(x='airport_name', y='prediction', xerr='err', kind='barh');


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[164]:

bestModelDF.printSchema()


# In[163]:

get_ipython().magic('matplotlib inline')


# In[162]:

sentiment_pd = best_model.    transform(testing_df).    groupby('airport_name').    agg(fn.avg('prediction').alias('prediction'), 
        (2*fn.stddev('prediction')/fn.sqrt(fn.count('*'))).alias('err')).\
    toPandas()


# In[ ]:

arr = ["emirates", "lufthansa", "etihad-airways", "qantas-airways", "jet-airways", "american-airlines", "ryanair", "air-canada-rouge", "united-airlines", "british-airways", "spirit-airlines"]


# In[ ]:

bestModelPD500 = sentiment_pd.loc[sentiment_pd['airline_name'].isin(arr)]


# In[ ]:

bestModelPD500


# In[ ]:

bestModelPD500.plot(x='airline_name', y='prediction', xerr='err', kind='barh');


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



