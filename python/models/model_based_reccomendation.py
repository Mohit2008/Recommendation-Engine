from surprise import Dataset, evaluate
from surprise import KNNBasic
from collections import defaultdict
import os, io
import numpy as np
from surprise import accuracy
import pandas as pd
from os import path
import logging
import sys
import time
from surprise.model_selection import cross_validate

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

here = path.abspath(path.dirname(__file__))
RECORD_FILE_PATH = os.path.join(here, os.pardir, "records", "../records/knn_based_recommendation.txt")
LOGGER = logging.getLogger(__name__)

sim_options = {
    'name': 'cosine',
    'user_based': False
}


def load_data(start_time):
    data = Dataset.load_builtin("ml-100k")
    print("Data loaded")
    trainingSet = data.build_full_trainset()
    LOGGER.info("0;Successfully imported movie lens data in '%s' minutes",
                str((time.time() - start_time) / 60))
    return trainingSet


def do_knn(trainingSet, start_time):
    knn = KNNBasic(sim_options=sim_options)
    # evaluate(knn, Dataset.load_builtin("ml-100k"), measures=['RMSE', 'MAE'])
    knn.fit(trainingSet)
    testSet = trainingSet.build_anti_testset()
    print("Training complete")
    predictions = knn.test(testSet)
    print("Predictions ready")
    LOGGER.info("0;Data prediction completed in '%s' minutes",
                str((time.time() - start_time) / 60))
    print("Rmse values for doing model based recomm on movielens data is " + str(accuracy.rmse(predictions)))
    return predictions


def get_top_recommendations(predictions, start_time):
    top_recs = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        top_recs[uid].append((iid, est))
    for uid, user_ratings in top_recs.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_recs[uid] = user_ratings[:100]
    LOGGER.info("0;Generated top recommendations and loaded into dataframe in '%s' minutes",
                str((time.time() - start_time) / 60))
    return top_recs


def read_item_names(start_time):
    """Read the u.item file from MovieLens 100-k dataset and returns a
    mapping to convert raw ids into movie names.
    """
    file_name = (os.path.expanduser('~') +
                 '/.surprise_data/ml-100k/ml-100k/u.item')
    rid_to_name = {}
    with io.open(file_name, 'r', encoding='ISO-8859-1') as f:
        for line in f:
            line = line.split('|')
            rid_to_name[line[0]] = line[1]
    LOGGER.info("0;Movie Item names imported in '%s' minutes", str((time.time() - start_time) / 60))
    return rid_to_name


def get_user_rating(userid, file, topk, start_time):
    LOGGER.info("0;Fetching the records for user '%s'", str(userid))
    input = pd.read_csv(file, sep="\t", header=None, index_col=0)
    if int(userid) in input.index:
        user_data = (input.loc[int(userid)][:int(topk)])
        print("For user id " + str(userid) + " top " + str(topk) + " predictions are: \n" + str(user_data))
        LOGGER.info("0;Records successfully fetched for user '%s' in '%s' minutes", str(userid),
                    str((time.time() - start_time) / 60))
    else:
        print("You have entered an invalid userid")
        LOGGER.info("0;A user with userid '%s' doesnt exist in our records", str(userid))


def is_user_valid(userid, file):
    input = pd.read_csv(file, sep="\t", header=None, index_col=0)
    if int(userid) in input.index:
        return True
    return False


def do_knn_on_movie_lens(uid, topk=5, refresh="no"):
    start_time = time.time()
    LOGGER.info("0; Starting KNN model based recommendation engine")
    if refresh == "no" and (os.path.isfile(RECORD_FILE_PATH)):
        LOGGER.info(
            "0;Since the refresh is not requested and a local copy of record is found hence rendering the response")
        print("Existing record found....generating recommendation from the records")
        get_user_rating(uid, RECORD_FILE_PATH, topk, start_time)
    else:
        if os.path.isfile(RECORD_FILE_PATH):
            LOGGER.info("0;Since the user requested new recommendations hence resubmitting the job")
            print("Generating new recommendation since the user requested refresh")
            if not is_user_valid(uid, RECORD_FILE_PATH):
                print("No such user found , enter a valid uid")
                return
        else:
            print("Generating new recommendation as no previous records found")
        trainingSet = load_data(start_time)
        predictions = do_knn(trainingSet, start_time)
        top_recommendations = get_top_recommendations(predictions, start_time)
        rid_to_name = read_item_names(start_time)
        if (os.path.isfile(RECORD_FILE_PATH)):
            os.remove(RECORD_FILE_PATH)
        with open(RECORD_FILE_PATH, "w") as text_file:
            for userid, user_ratings in top_recommendations.items():
                info = (userid, [rid_to_name[iid] for (iid, _) in user_ratings])
                items = "\t".join(info[1])
                output = info[0] + "\t" + items
                text_file.write(output.encode('utf-8'))
                text_file.write("\n")
        print("The recommendations have been generated")
        LOGGER.info("0;All recommendations generated are written to '%s' in '%s' minutes", str(RECORD_FILE_PATH),
                    str((time.time() - start_time) / 60))
        get_user_rating(uid, RECORD_FILE_PATH, topk, start_time)
