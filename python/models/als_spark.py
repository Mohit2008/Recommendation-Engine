from logging.handlers import WatchedFileHandler  # NOQA
import logging  # NOQA
import os  # NOQA
import sys  # NOQA
import zipfile
from pyspark.mllib.recommendation import ALS
from math import sqrt
from pyspark import SparkConf, SparkContext
import math
import time
from operator import add
# import urllib.request
import urllib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

current_path=os.path.dirname(__file__)
datasets_path = os.path.join(current_path,'../', 'data')
LOGGER = logging.getLogger(__name__)

def load_spark_context():
    conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationsALS")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir('checkpoint')
    return sc


def load_data(start_time):
    # small_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest-small.zip'
    complete_dataset_url = 'http://files.grouplens.org/datasets/movielens/ml-latest.zip'
    complete_dataset_path = os.path.join(datasets_path, 'ml-latest.zip')
    # small_dataset_path = os.path.join(datasets_path, 'ml-latest-small.zip')
    if not os.path.isfile(complete_dataset_path):
        # small_f = urllib.urlretrieve(small_dataset_url, small_dataset_path)
        complete_f = urllib.urlretrieve (complete_dataset_url, complete_dataset_path)
        # with zipfile.ZipFile(small_dataset_path, "r") as z:
        #     z.extractall(datasets_path)

        with zipfile.ZipFile(complete_dataset_path, "r") as z:
                z.extractall(datasets_path)
    ratings_file = os.path.join(datasets_path, 'ml-latest', 'ratings.csv')
    LOGGER.info("0;Successfully imported movie lens data in '%s' minutes",
                str((time.time() - start_time) / 60))
    return ratings_file


def transform_data(sc, small_ratings_file,start_time):
    small_ratings_raw_data = sc.textFile(small_ratings_file)
    small_ratings_raw_data_header = small_ratings_raw_data.take(1)[0]

    small_ratings_data = small_ratings_raw_data.filter(lambda line: line != small_ratings_raw_data_header) \
        .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1], tokens[2])).cache()

    small_movies_file = os.path.join(datasets_path, 'ml-latest', 'movies.csv')

    small_movies_raw_data = sc.textFile(small_movies_file)
    small_movies_raw_data_header = small_movies_raw_data.take(1)[0]

    small_movies_data = small_movies_raw_data.filter(lambda line: line != small_movies_raw_data_header) \
        .map(lambda line: line.split(",")).map(lambda tokens: (tokens[0], tokens[1])).cache()
    LOGGER.info("0;Data ready for prediction in '%s' minutes",
                str((time.time() - start_time) / 60))
    return small_ratings_data


def do_cross_validation(small_ratings_data, start_time):
    training_RDD, validation_RDD, test_RDD = small_ratings_data.randomSplit([6, 2, 2], seed=0)
    validation_for_predict_RDD = validation_RDD.map(lambda x: (x[0], x[1]))
    test_for_predict_RDD = test_RDD.map(lambda x: (x[0], x[1]))
    LOGGER.info("0;Data splitted into test-train in '%s' minutes",
                str((time.time() - start_time) / 60))
    return training_RDD, validation_for_predict_RDD, validation_RDD, test_for_predict_RDD




def do_als(training_RDD, validation_for_predict_RDD, validation_RDD, test_for_predict_RDD, items, start_time):
    seed = 5
    iterations = 10
    regularization_parameter = 0.1
    ranks = [4, 8, 12]
    errors = [0, 0, 0]
    err = 0
    tolerance = 0.02
    min_error = float('inf')
    best_rank = -1
    best_iteration = -1
    for rank in ranks:
        model = ALS.train(training_RDD, rank, seed=seed, iterations=iterations,
                          lambda_=regularization_parameter)
        predictions = model.predictAll(validation_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
        rates_and_preds = validation_RDD.map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))).join(predictions)
        error = math.sqrt(rates_and_preds.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())
        errors[err] = error
        err += 1
        print 'For rank %s the RMSE is %s' % (rank, error)
        if error < min_error:
            min_error = error
            best_rank = rank

    print('The best model was trained with rank %s' % best_rank)
    print(predictions.take(items))
    LOGGER.info("0;Predictions completed in '%s' minutes",
                str((time.time() - start_time) / 60))



def start_spark_recommendation(userid, items, refresh):
    start_time = time.time()
    small_ratings_file = load_data(start_time)
    sc = load_spark_context()
    small_ratings_data = transform_data(sc, small_ratings_file,start_time)
    training_RDD, validation_for_predict_RDD, validation_RDD, test_for_predict_RDD = do_cross_validation(
        small_ratings_data, start_time)
    do_als(training_RDD, validation_for_predict_RDD, validation_RDD, test_for_predict_RDD, items,start_time)
