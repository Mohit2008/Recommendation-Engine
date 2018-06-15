import pandas as pd
import numpy as np
from Cython.Plex.Regexps import RE
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
import os
from PIL import Image
from urllib2 import urlopen
import sys
import requests

from os import path
import argparse
import time

sys.path.append(os.path.join(os.path.abspath(os.path.dirname(__file__)), os.pardir, os.pardir))

here = path.abspath(path.dirname(__file__))


def get_data(start_time, input_file_path):
    data = pd.read_csv(input_file_path).set_index('user')
    print("Data import complete")
    elaps = (time.time() - start_time) / 60
    print("Successfully imported data from %s in %s minutes" % (input_file_path, elaps))
    return data


def do_predictions(data, start_time):
    data_matrix = data.as_matrix()
    user_ratings_mean = np.mean(data_matrix, axis=1)
    data_normalised = data_matrix - user_ratings_mean.reshape(-1, 1)
    elaps = (time.time() - start_time) / 60
    print("Data normalised in %s minutes" % str(elaps))
    U, sigma, Vt = svds(data_normalised, k=80)
    elaps = (time.time() - start_time) / 60
    print("Data successfully decomposed into 3 singular matrix in with 80 iterations in %s minutes"
          % str((elaps)))
    sigma = np.diag(sigma)
    all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
    predicted_df = pd.DataFrame(all_user_predicted_ratings, columns=data.columns, index=data.index)

    print("Rmse values for doing svd on transaction data is " + str(((data.sub(predicted_df).pow(2).mean()).mean())))
    elaps = (time.time() - start_time) / 60
    print("Data prediction completed and loaded into dataframe in %s minutes"
          % str(elaps))
    print("Done predicting the ratings for all users and all items")
    return predicted_df


def get_product_image(user_data):
    prod_links = {
        "red hot chili peppers": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ1XeuRyHbmHL_vXtm6zesptPN4UO3CSGy1Kb7IS0Hv9Mv2m0iT",
        "the killers": "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQcawUk4qsSmE6kDaZHh_jNrzblblcbCgJ4eprGkEOHQFJzzgmvyA",
        "jack johnson": "https://upload.wikimedia.org/wikipedia/en/thumb/4/42/Schandmaul-Leuchtfeuer.jpg/220px-Schandmaul-Leuchtfeuer.jpg"}
    if user_data in prod_links:
        image_url = prod_links[user_data]
        image = user_data + '.jpg'
        img_data = requests.get(image_url).content
        with open(image, 'wb') as handler:
            handler.write(img_data)
        return image


def merge_images(image_list):
    images = map(Image.open, image_list)
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset, 0))
        x_offset += im.size[0]
    new_im.save('user1_recommendations.jpg')
    for i in image_list:
        os.remove(i)


def get_user_rating(userid, file, topk, start_time):
    image_list = []
    print("Fetching the records for user %s" % str(userid))
    input = pd.read_csv(file, sep="\t", header=None, index_col=0)
    if int(userid) in input.index:
        user_data = str(input.loc[userid][:topk])
        user_data_list = list(input.loc[userid][:topk])
        print("For user id " + str(userid) + " top " + str(topk) + " predictions are: \n" + str(user_data))
        if userid == 1 and len(user_data_list) == 3:
            for i in user_data_list:
                image_list.append(get_product_image(i))
            merge_images(image_list)
        elaps = (time.time() - start_time) / 60
        print("Records successfully fetched for user %s in %s minutes" % (str(userid),
                                                                          str(elaps)))
    else:
        print("You have entered an invalid userid")
        print("A user with userid %s doesnt exist in our records" % str(userid))


def is_user_valid(userid, file):
    input = pd.read_csv(file, sep="\t", header=None, index_col=0)
    if int(userid) in input.index:
        return True
    return False


def generate_ratings(uid, input_file_path, output_records_path, topk=5, refresh="no"):
    uid = int(uid)
    topk = int(topk)
    start_time = time.time()
    print("Starting SVD based recommendation engine")
    if refresh == "no" and os.path.isfile(output_records_path):
        print(
            "Since the refresh is not requested and a local copy of record is found hence rendering the response")
        print("Existing record found....generating recommendation from the records")
        input = pd.read_csv(output_records_path, sep="\t", header=None)
        get_user_rating(uid, output_records_path, topk, start_time)
    else:
        if os.path.isfile(output_records_path):
            print("Since the user requested new recommendations hence resubmitting the job")
            if not is_user_valid(uid, output_records_path):
                print("No such user found , enter a valid uid")
                return
        else:
            print("Generating new recommendation as no previous records found")
        data = get_data(start_time, input_file_path)
        predicted_df = do_predictions(data, start_time)
        with open(output_records_path, "w") as text_file:
            for userid in data.index:
                user_row = data.loc[userid]
                non_predicted_items = list(data.columns[(user_row == 0)])
                user_prediction = predicted_df.loc[userid]
                for item in non_predicted_items:
                    user_prediction.filter(like=item)
                top_predictions = pd.DataFrame(user_prediction.sort_values(ascending=False))
                out = str(userid) + "\t" + "\t".join(list(top_predictions.index))
                text_file.write(out)
                text_file.write("\n")
        print("The recommendations have been generated")
        print("0;All recommendations generated are written to %s in %s minutes" % (str(output_records_path),
                                                                                   str((
                                                                                       time.time() - start_time) / 60)))
        get_user_rating(uid, output_records_path, topk, start_time)


if __name__ == '__main__':
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument('--input', help='Location of input transaction data', required=True)
        parser.add_argument('--output', help='Location of output recommendations', required=True)
        parser.add_argument('--userid', help='UserId for which you want to get recommendations', required=True)
        parser.add_argument('--n', help='No of top recommendations')
        parser.add_argument('--refresh', help='Whether you want new recommendations("yes"/"no")')
        args = parser.parse_args()

        input_file_path = args.input
        output_records_path = args.output
        generate_ratings(args.userid, input_file_path, output_records_path, args.n, args.refresh)
    except Exception as e:
        parser.print_help()
        print("Exception occurred due to ", e)


#python svd_sk.py --output /home/khanna/codebase/recommendation-engine/python/records/svd_based_recommendation.txt --input /home/khanna/codebase/recommendation-engine/python/data/transaction_data.csv  --userid 1 --n 5 --refresh "yes"
