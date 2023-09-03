from pyspark.sql import SparkSession
from pyspark.sql.functions import from_unixtime
from pyspark.sql import functions as F
import findspark
import sparknlp
from pyspark.sql.types import *
from sparknlp.annotator import *
from pyspark.ml import Pipeline
from pyspark.sql.functions import col, concat_ws, split, coalesce, avg, when, corr
import requests
import csv
import matplotlib.pyplot as plt
import numpy as np
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression
from pyspark.sql.functions import udf, corr
from pyspark.ml.evaluation import RegressionEvaluator

##### START ########
findspark.init()
spark = sparknlp.start()
sc = spark.sparkContext
#spark = SparkSession.builder \
#    .appName("MySparkApp") \
 #   .getOrCreate()

sc = spark.sparkContext

##### SET YOUR PATH ######
# file path to our dataset
file_path = "data_file.csv"

# Create initial spark dataframe from CSV
initial_df = spark.read.csv(file_path, header=True, inferSchema=True)

def preprocess_dataframe(df):
    # Columns that we will need for analysis
    columns_to_keep = ["id", 'body', "created_utc", "subreddit"]
    # filter out columns
    df = df.select(*columns_to_keep)
    # filter comments so they mention Tesla
    df = df.filter((df['body'].like('%Tesla%')) | (df['body'].like('%TSLA%')))
    # remove all null values in "created_utc" because we need to know the time
    df = df.dropna(subset=["created_utc"])
    # add column with dates by converting UTC-values
    df = df.withColumn('date', from_unixtime(df["created_utc"], format='yyyy-MM-dd'))
    # some of the values in "created_utc" were not nulls but also not timestamps, so we
    # remove nulls again now from "date"
    df = df.dropna(subset=["date"])
    # sort the dataframe by date
    df = df.orderBy(col("date").asc())
    # create df that has only comments and dates
    filtered_columns = ["body", "date"]
    df = df.select(filtered_columns)
    # remove punctuation and special characters
    df = df.withColumn("text", F.regexp_replace("body", "[^\sa-zA-Z0-9]", ""))
    # make all the comments lowercase
    df = df.withColumn("text", F.lower(df["text"]))
    # remove "body" column
    df = df.select("date", "text")
    return df


# This function will tokenize and then lemmatize the comments in the dataframe and make them ready for sentiment analysis.
def df_pipeline(df):
    # Initial step in creating pipeline
    documentAssembler = DocumentAssembler() \
    .setInputCol("text") \
    .setOutputCol("document")
    # Identifies sentences within text (document)
    sentenceDetector = SentenceDetector() \
    .setInputCols(["document"]) \
    .setOutputCol("sentence")
    # Toeknizer wil create a token from every in the comment. It is separated by space.
    tokenizer = Tokenizer() \
    .setInputCols(["sentence"]) \
    .setOutputCol("token")
    # Lemmatizer will lemmatize all the tokens and change them from for ex. "bulls" into "bull"
    lemmatizer = LemmatizerModel.pretrained() \
    .setInputCols(["token"]) \
    .setOutputCol("lemma")
    # Create the pipeline that will combine all the stages above
    pipeline = Pipeline() \
    .setStages([
    documentAssembler,
    sentenceDetector,
    tokenizer,
    lemmatizer
    ])
    # Create result dataframe that consists of elements from pipeline
    result = pipeline.fit(df).transform(df)
    # Extract result and date from the result and save it as a dataframe
    df = result.selectExpr("lemma.result", "date")
    return df

# This function will download the AFINN-111 sentiment dictionary file and save it locally.
def download_sentiment_dictionary():
    url = 'https://raw.githubusercontent.com/fnielsen/afinn/master/afinn/data/AFINN-111.txt'
    # Send a request to the URL to download the file
    response = requests.get(url)
    # Check if the request was successful
    if response.status_code == 200:
        # Save the contents as a file
        with open('AFINN-111.txt', 'w') as file:
            file.write(response.text)
        print('AFINN-111 dictionary downloaded successfully.')
    else:
        print('Failed to download the AFINN-111 dictionary.')


# this function will read the downloaded sentiment dictionary file and then update the words with new sentiment scores.
# if the word already exists in the ditcionary it will only update sentiment score, if the word doesn't exist it will add
# a new word.

def update_sentiment_dictionary():
    df = spark.read.text('AFINN-111.txt')
    df = df.withColumn('word', split(df['value'], '\t')[0])
    df = df.withColumn('score', split(df['value'], '\t')[1].cast('integer'))
    df = df.drop("value")

    # we will add new_words to the dictionary or overwrite the sentiment score of the existing ones
    new_words = [("lit", 3), ("green", 3), ("red", -3), ("buy", 3), ("bull", 4), ("bear", -4), ("sell", -3), ("moon", 4),
                 ("stonks", 3), ("flame", 3), ("fly", 3), ("bearish", -3), ("shorting", -2), ("short", -2), ("long", 2), ("expansion", 1),
                 ("slay", 3), ("king", 2), ("mooning", 4), ("drop", -3), ("overvalue", -3), ("top", 2), ("kill", 2), ("pump", 2),
                 ("dump", -2), ("put", -2), ("bullish", 4), ("peak", -2), ("invest", 2), ("call", 2), ("retarded", -2), ("scam", -4)]

    # Create a dictionary from the DataFrame
    rows = df.collect()
    dictionary = {row["word"]: row["score"] for row in rows}
    for word, score in new_words:
        if word in dictionary:
            dictionary[word] = score  # Update sentiment score
        else:
            dictionary[word] = score  # Add new word and sentiment score

    # Convert updated dictionary back to DataFrame
    updated_dictionary_df = spark.createDataFrame([(word, str(score)) for word, score in dictionary.items()])
    updated_dictionary_df = updated_dictionary_df.withColumnRenamed("_1", "word")
    updated_dictionary_df = updated_dictionary_df.withColumnRenamed("_2", "score")
    # Create an RDD from dataframe
    rdd = updated_dictionary_df.rdd.map(lambda row: " ".join([str(row.word), str(row.score)]))
    # sort the rdd
    rdd = rdd.sortBy(lambda x: x[0])
    # ensure that saved txt file with consist of 1 file
    rdd = rdd.coalesce(1)
    # save the updated dictionary file localy
    rdd.saveAsTextFile("AFINN-112.txt")

# Function that will create a sentiment dataframe from the dictionary that we created. It will consist from words
# and their sentiment scores
def create_sentiment_df():
    sentiment_df = spark.read.csv("AFINN-112.txt/part-00000", header=True, inferSchema=True, sep=" ")
    column_names = ['word', 'score']  # Replace with your actual column names
    sentiment_df = sentiment_df.toDF(*column_names)
    return sentiment_df


# This function will calculate the sentiment score for every comment. It takes dataframe with comments and dataframe
# with words and scores as arguments
def calculate_sentiment(df, sentiment_df):
    # in order to make function work we need to turn every line into a single string
    df = df.withColumn("result_str", concat_ws(" ", col("result")))
    # We perform left join in order to find matching words in both dataframes with their sentiment score
    df = df.join(sentiment_df, col("result_str").contains(col("word")), "left")
    # We calculate the average sentiment score per sentence
    df = df.groupBy("result_str", "date").agg(avg(col("score")).alias("sentiment_score"))
    # If the sentiment score is < 0.1 or > -0.1 then the comment will be considered to be neutral
    df = df.withColumn("sentiment_polarity", when(
        col("sentiment_score") > 0.1, "Positive"
    ).when(
        col("sentiment_score") < -0.1, "Negative"
    ).otherwise("Neutral"))
# remove possible null values
    df = df.dropna()
# finally order the dataframe by date
    df = df.orderBy(col("date").asc())
    return df

# Fucntion to calculate average sentiment per day
def sentiment_per_day(df):
    sentiment_per_day = df.groupBy("date").agg(F.avg("sentiment_score").alias("average_score"))
    sentiment_per_day = sentiment_per_day.dropna()
    return sentiment_per_day


# We are using Alpha Vantage API to access the stock data. In order to send the request to Alpha Vantage API we need to specify
# the type of function in our case it is TIME_SERIES_DAILY_ADJUSTED, ticker (stock name abbreviation) in our case it TSLA,
# API key, datatype in our case it's CSV and outputsize which means how many data entries we want to get, by default it is 100
# but we will download the whole dataset by specifying outputsize=full.
def download_stock_data(ticker, start_date, end_date):
    url = "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED&symbol=" + ticker + "&apikey=XBJLPVXJGR3RZOH7&datatype=csv&outputsize=full"
    # send the request
    r = requests.get(url)
    # decode the collected data
    decoded_content = r.content.decode('utf-8')
    # write it into a csv
    cr = csv.reader(decoded_content.splitlines(), delimiter=',')
    data_list = list(cr)
    # print(data_list)
    df_stock = spark.createDataFrame(data_list[1:], schema=data_list[0])
    columns_to_keep = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_stock = df_stock.select(columns_to_keep)
    # filter out data that we are not interested at based on the dates
    df_stock = df_stock.filter((col('timestamp') >= start_date) & (col('timestamp') <= end_date))
    # add a column that shows how changed the stock price at the end of the day
    df_stock = df_stock.withColumn('difference', col('close') - col('open'))
    # rename column timestamp into date so we can later merge dataframes
    df_stock = df_stock.withColumnRenamed('timestamp', 'date')
    # sort dataframe into ascending order by date
    df_stock = df_stock.orderBy(col("date").asc())
    return(df_stock)


# Function to create a dataframe that will contain daily stock data and daily sentiment score
def combine_stocks_sentiment(df_stock, sentiment_per_day):
    # Merge two datafrmaes on the column "date"
    sentiment_stocks = df_stock.join(sentiment_per_day, on='date', how='inner')
    sentiment_stocks.show()
    # change the types of high and low from str to DoubleType so we can later calculate the average price during the day
    sentiment_stocks = sentiment_stocks.withColumn('high', col('high').cast(DoubleType()))
    sentiment_stocks = sentiment_stocks.withColumn('low', col('low').cast(DoubleType()))
    return sentiment_stocks

# UDF function that will calculate the average price by summing up high and low prices and deviding by 2
average_udf = udf(lambda x, y: (x+y)/2, DoubleType())

# Function to build a linear regression model. This model should be sufficient for our purposes because the
# dependence between average sentiment score and average stock prices or difference between opening and closing
# should be linear. Howeveer due to extremely low correlation this model won't perform well.
def linear_regression_model(df):
    assembler = VectorAssembler(inputCols=['average_score'], outputCol='features')
    feature_vector = assembler.transform(df).select('features', 'average_price')
    # Split the data into training and testing sets
    (training_data, test_data) = feature_vector.randomSplit([0.8, 0.2], seed = 111)
    # Create a LinearRegression model
    lr = LinearRegression(labelCol='average_price')
    # Fit the model to the training data
    model = lr.fit(training_data)
    # Make predictions on the test data
    predictions = model.transform(test_data)
    return predictions, model, test_data

def evaluate_model(model, test_data):
    evaluation = model.evaluate(test_data)
    evaluator = RegressionEvaluator(labelCol='average_price')
    # Calculate metrics
    r2 = evaluation.r2
    mae = evaluator.evaluate(predictions, {evaluator.metricName: 'mae'})
    rmse = evaluator.evaluate(predictions, {evaluator.metricName: 'rmse'})
    mse = evaluator.evaluate(predictions, {evaluator.metricName: 'mse'})
    # Print the metrics
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('MSE:', mse)
    print('R-squared:', r2)

# We are using pandas for plotting purposes, since it is only a visualisation it doesn't have to be scalable.
def plot_model_results_average_price(predictions, test_data):
    # Convert the 'predictions' DataFrame to a Pandas DataFrame
    predictions_pd = predictions.select('prediction').toPandas()
    # Convert the 'test_data' DataFrame to a Pandas DataFrame
    test_data_pd = test_data.select('average_price').toPandas()
    # Create an array of indices that will represent the time seires on which we will plot real and predicted stock dynamics
    indices = np.arange(len(predictions_pd))
    # Plot the predicted and real values
    plt.figure(figsize=(8, 6))
    plt.plot(indices, predictions_pd['prediction'], color='red', label='Predicted')
    plt.plot(indices, test_data_pd['average_price'], color='blue', label='Real')
    plt.xlabel('Index')
    plt.ylabel('Average Price')
    plt.title('Predicted vs. Real Values')
    plt.legend()
    plt.show()


######## CALLING FUNCTIONS ##########
df = preprocess_dataframe(initial_df)
df.show()

df = df_pipeline(df)
df.show()

download_sentiment_dictionary()
update_sentiment_dictionary()

sentiment_df = create_sentiment_df()
sentiment_df.show()

df = calculate_sentiment(df, sentiment_df)
df.show()

sentiment_per_day = sentiment_per_day(df)
# stocks we are interested at
ticker = 'TSLA'
#specify start and end date so we can filter out data that we are not interested at
start_date = "2020-02-22"
end_date = "2021-02-16"

df_stock = download_stock_data(ticker, start_date, end_date)
df_stock.show()

df = combine_stocks_sentiment(df_stock, sentiment_per_day)
#df.show()

# Correlation between average sentiment score and stock price change
correlation = df.select(corr('difference', 'average_score')).first()[0]
print("The correlation  average sentiment score and stock price change is: ", correlation)
# As expected the correlation is positive however it is very weak which will cause problems for model building

# add an average price column to the dataframe
df = df.withColumn("average_price", average_udf(df.high, df.low))
df.show()

# assign variables to linear regression model output
predictions, model, test_data = linear_regression_model(df)
predictions.show()

# Show the model evaluation results. As we can see due to insignificant correlation the performance of prediction
# model is unsatisfying. There is a number of reasons that could be a reason for this underperformance. First of all
# lack of the comments per day, there are some days that have only 1 or 2 comments mentioning Tesla and predicting
# stock movements based on it is nearly impossible. The second reason is the insuffiecient sentiment analysis, a general
# rule based approach even with addtional specific lexicon and adjusted sentiment didn't show a good performance on
# sentiment analysis.

evaluate_model(model, test_data)

# Of course visualising a model after an observed underperformance doesn't make much sense, however it still could be
# interesting to illustrate that because of the insignificant correlation the predicted values barely differtiate
# from the mean.
