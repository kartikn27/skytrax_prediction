
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


# In[3]:

import pandas as pd


# In[17]:

training, validation, testing = airlineDf.randomSplit([0.6, 0.3, 0.1], seed=0)


# In[36]:

get_ipython().system('curl https://raw.githubusercontent.com/quankiquanki/skytrax-reviews-dataset/master/data/airport.csv?accessType=DOWNLOAD > airport.csv')

get_ipython().system('hdfs dfs -put airport.csv   ')

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

airportCleanDF = reduce(DataFrame.drop, ['overall_rating','queuing_rating', 'terminal_cleanliness_rating', 'terminal_seating_rating', 'terminal_signs_rating', 'food_beverages_rating', 'airport_shopping_rating', 'wifi_connectivity_rating', 'airport_staff_rating','recommended'], airportCleanDF)


# In[7]:

from pyspark.ml.feature import Tokenizer
tokenizer = Tokenizer().setInputCol('content').setOutputCol('words')


# In[8]:

airlineCleanDF = airlineCleanDF.na.drop(subset=["content"]) # Remove rows with NULL in column 'content'
tokenizer.transform(airlineCleanDF)


# In[7]:

#tokenizer.transform(airlineCleanDF).show(5)


# In[5]:

from pyspark.ml.feature import CountVectorizer


# In[9]:

count_vectorizer_estimator = CountVectorizer().setInputCol('words').setOutputCol('features')
count_vectorizer_transformer = count_vectorizer_estimator.fit(tokenizer.transform(airlineCleanDF))


# In[10]:

#count_vectorizer_transformer.transform(tokenizer.transform(airlineCleanDF)).show(truncate=False)


# In[11]:

#count_vectorizer_transformer.vocabulary


# In[12]:

from pyspark.ml import Pipeline


# In[13]:

pipeline_cv_estimator = Pipeline(stages=[tokenizer, count_vectorizer_estimator])
pipeline_cv_transformer = pipeline_cv_estimator.fit(airlineCleanDF)
#pipeline_cv_transformer.transform(airlineCleanDF).show()


# In[14]:

sentiments_df = spark.read.parquet('/datasets/sentiment_analysis/sentiments.parquet')


# In[15]:

sentiments_df.printSchema()


# In[16]:

from pyspark.ml.feature import RegexTokenizer
tokenizer = RegexTokenizer().setGaps(False)  .setPattern("\\p{L}+")  .setInputCol("content")  .setOutputCol("words")


# In[17]:

content_words_df = tokenizer.transform(airlineCleanDF)


# In[18]:

#content_words_df.show(5)


# In[19]:

content_words_df.select('airline_name', fn.explode('words').alias('word')).show(5)


# In[20]:

content_word_sentiment_df = content_words_df.    select('airline_name', fn.explode('words').alias('word')).    join(sentiments_df, 'word')


# In[21]:

simple_sentiment_prediction_df = content_word_sentiment_df.    groupBy('airline_name').    agg(fn.avg('sentiment').alias('avg_sentiment')).    withColumn('predicted', fn.when(fn.col('avg_sentiment') > 0, 1.0).otherwise(0.))


# In[22]:

simple_sentiment_prediction_df.show(5)


# In[23]:

#airlineCleanDF.groupby('recommendedi').agg(fn.count('*')).show()


# In[24]:

airlineCleanDF.    join(simple_sentiment_prediction_df, 'airline_name').    select(fn.expr('float(recommendedi = predicted)').alias('correct')).    select(fn.avg('correct')).    show()


# In[25]:

#### DATA DRIVEN APPROACH
#### DATA DRIVEN APPROACH
#### DATA DRIVEN APPROACH


# In[26]:

# we obtain the stop words from a website
import requests
stop_words = requests.get('http://ir.dcs.gla.ac.uk/resources/linguistic_utils/stop_words').text.split()
len(stop_words)


# In[27]:

from pyspark.ml.feature import StopWordsRemover
sw_filter = StopWordsRemover()  .setStopWords(stop_words)  .setCaseSensitive(False)  .setInputCol("words")  .setOutputCol("filtered")


# In[28]:

from pyspark.ml.feature import CountVectorizer

# we will remove words that appear in 5 docs or less
cv = CountVectorizer(minTF=1., minDF=5., vocabSize=2**17)  .setInputCol("filtered")  .setOutputCol("tf")


# In[29]:

# we now create a pipelined transformer
cv_pipeline = Pipeline(stages=[tokenizer, sw_filter, cv]).fit(airlineCleanDF)


# In[30]:

# now we can make the transformation between the raw text and the counts
#cv_pipeline.transform(airlineCleanDF).show(5)


# In[31]:

from pyspark.ml.feature import IDF
idf = IDF().    setInputCol('tf').    setOutputCol('tfidf')


# In[32]:

idf_pipeline = Pipeline(stages=[cv_pipeline, idf]).fit(airlineCleanDF)


# In[33]:

#idf_pipeline.transform(airlineCleanDF).show(5)


# In[34]:

tfidf_df = idf_pipeline.transform(airlineCleanDF)


# In[35]:

tfidf_df.printSchema()


# In[36]:

### DATA SCIENCE PIPELINE FOR ESTIMATING SENITMENTS
### DATA SCIENCE PIPELINE FOR ESTIMATING SENITMENTS
### DATA SCIENCE PIPELINE FOR ESTIMATING SENITMENTS


# In[37]:

training_df, validation_df, testing_df = airlineCleanDF.randomSplit([0.6, 0.3, 0.1], seed=0)


# In[38]:

[training_df.count(), validation_df.count(), testing_df.count()]


# In[39]:

#  Logistic Regression
from pyspark.ml.classification import LogisticRegression


# In[40]:

lr = LogisticRegression().    setLabelCol('recommendedi').    setFeaturesCol('tfidf').    setRegParam(0.0).    setMaxIter(100).    setElasticNetParam(0.)


# In[41]:

lr_pipeline = Pipeline(stages=[idf_pipeline, lr]).fit(training_df)


# In[42]:

lr_pipeline.transform(validation_df).    select(fn.expr('float(prediction = recommendedi)').alias('correct')).    select(fn.avg('correct')).show()


# In[43]:

#airlineCleanDF.groupby('airline_name').agg(fn.count('*')).show()


# In[44]:

import pandas as pd
vocabulary = idf_pipeline.stages[0].stages[-1].vocabulary
weights = lr_pipeline.stages[-1].coefficients.toArray()
coeffs_df = pd.DataFrame({'word': vocabulary, 'weight': weights})


# In[45]:

coeffs_df.sort_values('weight').head(5)


# In[46]:

coeffs_df.sort_values('weight', ascending=False).head(5)


# In[47]:

lambda_par = 0.02
alpha_par = 0.3
en_lr = LogisticRegression().        setLabelCol('recommendedi').        setFeaturesCol('tfidf').        setRegParam(lambda_par).        setMaxIter(100).        setElasticNetParam(alpha_par)


# In[48]:

en_lr_estimator = Pipeline(
    stages=[tokenizer, sw_filter, cv, idf, en_lr])


# In[49]:

en_lr_pipeline = en_lr_estimator.fit(training_df)


# In[50]:

en_lr_pipeline.transform(validation_df).select(fn.avg(fn.expr('float(prediction = recommendedi)'))).show()


# In[51]:

en_weights = en_lr_pipeline.stages[-1].coefficients.toArray()
en_coeffs_df = pd.DataFrame({'word': en_lr_pipeline.stages[2].vocabulary, 'weight': en_weights})


# In[52]:

en_coeffs_df.sort_values('weight').head(15)


# en_coeffs_df.sort_values('weight').head(15)

# In[53]:

en_coeffs_df.sort_values('weight', ascending=False).head(15)


# In[54]:

en_coeffs_df.query('weight == 0.0').shape


# In[55]:

en_coeffs_df.query('weight == 0.0').shape[0]/en_coeffs_df.shape[0]


# In[56]:

en_coeffs_df.query('weight == 0.0').head(15)


# In[57]:

from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit


# In[58]:

en_lr_estimator.getStages()


# In[59]:

grid = ParamGridBuilder().    addGrid(en_lr.regParam, [0., 0.01, 0.02]).    addGrid(en_lr.elasticNetParam, [0., 0.2, 0.4]).    build()


# In[60]:

grid


# In[61]:

all_models = []
for j in range(len(grid)):
    print("Fitting model {}".format(j+1))
    model = en_lr_estimator.fit(training_df, grid[j])
    all_models.append(model)


# In[62]:

# estimate the accuracy of each of them:
accuracies = [m.    transform(validation_df).    select(fn.avg(fn.expr('float(recommendedi = prediction)')).alias('accuracy')).    first().    accuracy for m in all_models]


# In[63]:

accuracies


# In[64]:

import numpy as np


# In[65]:

best_model_idx = np.argmax(accuracies)


# In[66]:

grid[best_model_idx]


# In[67]:

best_model = all_models[best_model_idx]


# In[68]:

accuracies[best_model_idx]


# In[69]:

bestModelDF = best_model.transform(airlineCleanDF)


# In[70]:

bestModelDF.printSchema()


# In[71]:

bestModelDF.select('airline_name', 'content', 'prediction').show(30)


# In[73]:

# From best fit model select only the airlines where review count is greater than 500

bestModelDF.groupBy('airline_name').    agg(fn.count('airline_name').alias('airline_review_count')).    where(fn.col('airline_review_count')>500).    orderBy(fn.desc('airline_review_count')).show()


# In[136]:

# spirit = bestModelDF.where(fn.col('airline_name')=="spirit-airlines")
# british = bestModelDF.where(fn.col('airline_name')=="british-airways")
# united = bestModelDF.where(fn.col('airline_name')=="united-airlines")
# jet = bestModelDF.where(fn.col('airline_name')=="jet-airways")
# canada = bestModelDF.where(fn.col('airline_name')=="air-canada-rouge")
# ryanair = bestModelDF.where(fn.col('airline_name')=="ryanair")
# american = bestModelDF.where(fn.col('airline_name')=="american-airlines")
# lufthansa = bestModelDF.where(fn.col('airline_name')=="lufthansa")
# qantas = bestModelDF.where(fn.col('airline_name')=="qantas-airways")
# etihad = bestModelDF.where(fn.col('airline_name')=="etihad-airways")
# emirates = bestModelDF.where(fn.col('airline_name')=="emirates")


# In[137]:

# def unionAll(*dfs):
#     return reduce(DataFrame.unionAll, dfs)


# In[138]:

#bestModelDF500 = unionAll(spirit, british, united, jet, canada, ryanair, american, lufthansa, qantas, etihad, emirates)


# In[140]:

#bestModelDF500.count()


# In[74]:

get_ipython().magic('matplotlib inline')


# In[75]:

import seaborn


# In[77]:

sentiment_pd = best_model.    transform(airlineCleanDF).    groupby('airline_name').    agg(fn.avg('prediction').alias('prediction'), 
        (2*fn.stddev('prediction')/fn.sqrt(fn.count('*'))).alias('err')).\
    toPandas()


# In[78]:

arr = ["emirates", "lufthansa", "etihad-airways", "qantas-airways", "jet-airways", "american-airlines", "ryanair", "air-canada-rouge", "united-airlines", "british-airways", "spirit-airlines"]


# In[79]:

bestModelPD500 = sentiment_pd.loc[sentiment_pd['airline_name'].isin(arr)]


# In[80]:

bestModelPD500


# In[81]:

bestModelPD500.plot(x='airline_name', y='prediction', xerr='err', kind='barh');


# In[ ]:




# In[48]:

imdb_reviews_df = spark.read.parquet('/datasets/sentiment_analysis/imdb_reviews_preprocessed.parquet')


# In[50]:

imdb_reviews_df.count()

