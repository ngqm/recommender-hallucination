"""
File name: preprocess.py
Author: Quang Minh Nguyen
Last update: 2024-12-13

Preprocessing script for the Amazon dataset.
"""


import json


def load_data(dataset):
    """Load reviews and metadata of a dataset.
    """
    if dataset == "All_Beauty":
        beauty_reviews = []
        with open('data/All_Beauty.jsonl', 'r') as file:
            for line in file:
                beauty_reviews.append(json.loads(line.strip()))
        beauty_meta = []
        with open('data/meta_All_Beauty.jsonl', 'r') as file:
            for line in file:
                beauty_meta.append(json.loads(line.strip()))
        return beauty_reviews, beauty_meta
    
    elif dataset == "Movies_and_TV":
        movies_reviews = []
        count = 0
        with open('data/Movies_and_TV.jsonl', 'r') as file:
            for line in file:
                if count >= 50000:
                    break
                movies_reviews.append(json.loads(line.strip()))
                count += 1
        relevant_movies = set(review['parent_asin'] for review in movies_reviews)
        movies_meta = []
        movies_so_far = set()
        with open('data/meta_Movies_and_TV.jsonl', 'r') as file:
            for line in file:
                meta = json.loads(line.strip())
                if not meta['parent_asin'] in relevant_movies:
                    continue
                movies_meta.append(meta)
                movies_so_far.add(meta['parent_asin'])
                # break if all movies in movies_reviews are found
                if len(movies_so_far) == len(relevant_movies):
                    break
        return movies_reviews, movies_meta
    

def get_core_data(reviews, meta):
    """
    Get reviews of users who have at least 5 and at most 20 reviews.
    Retain only relevant metadata in a processed set.

    :param reviews: list of reviews
    :param meta: list of metadata
    """

    reviews_set = {}
    for review in reviews:
        user_id = review['user_id']
        if user_id not in reviews_set:
            reviews_set[user_id] = []
        reviews_set[user_id].append(review)
    reviews_set = {k: v for k, v in reviews_set.items() if len(v) >= 5 and len(v) <= 20}
    
    product_set = set()
    for user_id, reviews in reviews_set.items():
        for review in reviews:
            product_set.add(review['parent_asin'])
    meta_processed = [meta for meta in meta if meta['parent_asin'] in product_set]
    return reviews_set, meta_processed


def save_processed_data(dataset):
    """
    Produce processed data for a dataset.

    :param dataset: name of the dataset. Either "All_Beauty" or "Movies_and_TV"
    """

    reviews, meta = load_data(dataset)
    reviews_set, meta_processed = get_core_data(reviews, meta)
    
    # put reviews_set into a jsonl file
    with open(f'data/{dataset}_processed.jsonl', 'w') as file:
        for user_id, reviews in reviews_set.items():
            for review in reviews:
                file.write(json.dumps(review) + '\n')
    # put meta_processed into a jsonl file
    with open(f'data/meta_{dataset}_processed.jsonl', 'w') as file:
        for meta in meta_processed:
            file.write(json.dumps(meta) + '\n')
    
    return reviews_set, meta_processed


def get_top_users(dataset, reviews_set, num_users):
    """
    Get users with the most reviews.

    :param reviews_set: processed data
    :param num_users: number of users to get
    """
    top_users = sorted(reviews_set.keys(), key=lambda x: len(reviews_set[x]), reverse=True)[:num_users]
    with open(f'data/{dataset}_test_users.txt', 'w') as file:
        [file.write(user_id + '\n') for user_id in top_users]
    return top_users


if __name__ == "__main":

    # produce processed data for All_Beauty
    beauty_user_reviews_set, _ = save_processed_data("All_Beauty")
    # produce processed data for Movies_and_TV
    movies_user_reviews_set, _ = save_processed_data("Movies_and_TV")

    # get 500 users with the most reviews in beauty_user_reviews_set
    top_beauty_users = get_top_users("All_Beauty", beauty_user_reviews_set, 500)
    print(f"Average number of reviews per user (beauty): {sum(len(beauty_user_reviews_set[user_id]) for user_id in top_beauty_users) / 500}")
    # get 500 users with the most reviews in movies_user_reviews_set
    top_movies_users = get_top_users("Movies_and_TV", movies_user_reviews_set, 500)
    print(f"Average number of reviews per user (movies): {sum(len(movies_user_reviews_set[user_id]) for user_id in top_movies_users) / 500}")
