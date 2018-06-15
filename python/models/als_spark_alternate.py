from logging.handlers import WatchedFileHandler  # NOQA
import logging  # NOQA
import os  # NOQA
import sys  # NOQA
import zipfile
from pyspark.mllib.recommendation import ALS
import time
from pyspark import SparkConf, SparkContext
import math
# import urllib.request
import urllib

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
LOGGER = logging.getLogger(__name__)
datasets_path = os.path.join('..', 'data')


def load_spark_context():
    conf = SparkConf().setMaster("local[*]").setAppName("MovieRecommendationsALS")
    sc = SparkContext(conf=conf)
    sc.setCheckpointDir('checkpoint')
    return sc

def load_movie_lens_rating_data(sc):
    complete_ratings_file = os.path.join(datasets_path, 'ml-latest-small', 'ratings.csv')
    complete_ratings_raw_data = sc.textFile(complete_ratings_file)
    complete_ratings_raw_data_header = complete_ratings_raw_data.take(1)[0]

    # Parse
    complete_ratings_data = complete_ratings_raw_data.filter(lambda line: line != complete_ratings_raw_data_header) \
        .map(lambda line: line.split(",")).map(
        lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))).cache()

    print "There are %s recommendations in the complete dataset" % (complete_ratings_data.count())
    LOGGER.info("0;Successfully imported movie lens data in '%s' minutes",
                str((time.time() - start_time) / 60))
    return complete_ratings_data


def load_movie_lens_movie_data(sc):
    complete_movies_file = os.path.join(datasets_path, 'ml-latest-small', 'movies.csv')
    complete_movies_raw_data = sc.textFile(complete_movies_file)
    complete_movies_raw_data_header = complete_movies_raw_data.take(1)[0]
    # Parse
    complete_movies_data = complete_movies_raw_data.filter(lambda line: line != complete_movies_raw_data_header) \
        .map(lambda line: line.split(",")).map(lambda tokens: (int(tokens[0]), tokens[1], tokens[2])).cache()

    complete_movies_titles = complete_movies_data.map(lambda x: (int(x[0]), x[1]))

    print "There are %s movies in the complete dataset" % (complete_movies_titles.count())
    return complete_movies_titles


def get_counts_and_averages(ID_and_ratings_tuple):
    nratings = len(ID_and_ratings_tuple[1])
    return ID_and_ratings_tuple[0], (nratings, float(sum(x for x in ID_and_ratings_tuple[1]))/nratings)


def do_spark_als():
    sc=load_spark_context()
    complete_ratings_data= load_movie_lens_rating_data(sc)
    complete_movies_titles= load_movie_lens_movie_data(sc)


    movie_ID_with_ratings_RDD = (complete_ratings_data.map(lambda x: (x[1], x[2])).groupByKey())
    movie_ID_with_avg_ratings_RDD = movie_ID_with_ratings_RDD.map(get_counts_and_averages)
    movie_rating_counts_RDD = movie_ID_with_avg_ratings_RDD.map(lambda x: (x[0], x[1][0]))

    new_ratings_model = ALS.train(complete_ratings_data, 4, seed=5,
                                  iterations=10, lambda_=0.1)
    new_user_recommendations_RDD = new_ratings_model.predictAll(complete_ratings_data.map(lambda x: (x[0], x[1])))


    # Transform new_user_recommendations_RDD into pairs of the form (Movie ID, Predicted Rating)
    new_user_recommendations_rating_RDD = new_user_recommendations_RDD.map(lambda x: (x.product, x.rating))
    new_user_recommendations_rating_title_and_count_RDD = \
        new_user_recommendations_rating_RDD.join(complete_movies_titles).join(movie_rating_counts_RDD)
    new_user_recommendations_rating_title_and_count_RDD.take(3)


    new_user_recommendations_rating_title_and_count_RDD = \
        new_user_recommendations_rating_title_and_count_RDD.map(lambda r: (r[1][0][1], r[1][0][0], r[1][1]))
    top_movies = new_user_recommendations_rating_title_and_count_RDD.filter(lambda r: r[2]>=25).takeOrdered(25, key=lambda x: -x[1])

    print ('TOP recommended movies (with more than 25 reviews):\n%s' %
            '\n'.join(map(str, top_movies)))
