import json
import re
import numpy as np

from transformers import pipeline
from Levenshtein import distance


def get_data(dataset):
    """
    Get review data, meta-data, train users, and test users for the given dataset.

    :param dataset: str, dataset name. Either "All_Beauty" or "Movies_and_TV"
    :return: review data, meta-data, train users, test users
    :rtype: list, list, list, list
    """

    # review data
    reviews = [
        json.loads(review) for review in open(f'../data/{dataset}_processed.jsonl', 'r')
    ]  # 10776
    meta = [
        json.loads(meta) for meta in open(f'../data/meta_{dataset}_processed.jsonl', 'r')
    ]  # 6170
    # train users
    train_users = []
    # with open(f'../data/{dataset}_train_users.txt', 'r') as f:
    #     train_users = f.read().splitlines()  # 1000
    # test users
    with open(f'../data/{dataset}_test_users.txt', 'r') as f:
        test_users = f.read().splitlines()  # 500
    return reviews, meta, train_users, test_users


# title, brand, categories (if any), description (if any), price (if any) of product
def get_string_from_meta(m_list):
    """
    Get the string representation of the metadata of the products in the list.

    :param m_list: list of metadata of products

    :return: list of string representation of the metadata of the products
    """
    info_list = []
    for m in m_list:
        info = ""
        info += f"Title: {m['title']}\n"  # title
        # brand for All_Beauty, 
        # director, actor, & studio for Movies
        for key in ['Brand', 'Director', 'Actors', 'Studio']:
            if key in m['details']:
                info += f"{key}: {m['details'][key]}\n"
        try:  # categories
            if len(m['categories']) > 0:
                info += f"Categories: {m['categories']}\n"
        except:
            pass
        try:  # description
            if len(m['description']) > 0:
                info += f"Description: {m['description']}\n"
        except:
            pass
        if m['price'] is not None:  # price
            info += f"Item Price: {m['price']}\n"
        info_list.append(info)
    
    return info_list


def get_embedding(meta):
    """Get embeddings of the given meta-data using Llama-3.2-1b model.

    :param meta: list, list of meta-data
    """
    extractor = pipeline(model="meta-llama/Llama-3.2-1B", task="feature-extraction", device='cuda:0')
    embeddings =extractor(get_string_from_meta(meta), return_tensors=True)
    return embeddings


def get_candidates(reviews, meta, user_id):
    """
    Get 20 candidate products for the user with user_id, among which
    1 is the new item in the user's purchase history and 19 are randomly sampled
    from products that the user has not purchased.

    :param reviews: list of reviews
    :param meta: list of metadata of products
    :param user_id: user id

    :return: list of parent_asin of 20 candidate products
    :rtype: list
    """
    # get all reviews associated with the user
    user_reviews = [r for r in reviews if r['user_id'] == user_id]
    # sort reviews by timestamp
    user_reviews.sort(key=lambda x: x['timestamp'])

    # get the new item in the user's purchase history
    new_item = user_reviews[-1]['parent_asin']

    # get all products that the user has not purchased
    not_purchased = [m['parent_asin'] for m in meta if m['parent_asin'] != new_item and m['parent_asin'] not in [r['parent_asin'] for r in user_reviews]]

    # sample 19 products from not_purchased
    np.random.seed(0)
    candidates = np.random.choice(not_purchased, 19).tolist() + [new_item]

    return candidates


def get_profile(shortname, profile_type, user_id):

    if profile_type == "iterative":
        with open(f"../profiles/{shortname}/iterative/user_{user_id}/user_{user_id}_target.json", 'r') as f:
            last_interaction = int(json.load(f)["interaction_number"])
        with open(f"../profiles/{shortname}/iterative/user_{user_id}/user_{user_id}_update_{last_interaction-1}.json", 'r') as f:
            user_profile = json.load(f)["profile"]
    elif profile_type == "preference_positive":
        with open(f"../profiles/{shortname}/preference/user_{user_id}.json", 'r') as f:
            user_profile_overall = json.load(f)["raw_response"]
            # parse the user profile with "POSITIVE SUMMARY:", "NEGATIVE SUMMARY:", and "OVERALL SUMMARY"
            user_profile = user_profile_overall[
                re.search("POSITIVE SUMMARY:", user_profile_overall).end():
                re.search("NEGATIVE SUMMARY:", user_profile_overall).start()]
    elif profile_type == "preference_negative":
        with open(f"../profiles/{shortname}/preference/user_{user_id}.json", 'r') as f:
            user_profile_overall = json.load(f)["raw_response"]
            # parse the user profile with "POSITIVE SUMMARY:", "NEGATIVE SUMMARY:", and "OVERALL SUMMARY"
            user_profile = user_profile_overall[
                re.search("NEGATIVE SUMMARY:", user_profile_overall).end():
                re.search("OVERALL SUMMARY:", user_profile_overall).start()]
    else:
        with open(f"../profiles/{shortname}/{profile_type}/user_{user_id}.json", 'r') as f:
            user_profile = json.load(f)["raw_response"]

    return user_profile


def get_ground_truth(shortname, user_id):
    with open(f"../profiles/{shortname}/vanilla/user_{user_id}.json", 'r') as f:
        ground_truth = json.load(f)["target_interaction"]["parent_asin"]
    return ground_truth


def get_validated_products(reviews, meta, user_id, recommendations):

    new_product_ids = []
    non_existing_product_count = 0
    candidates = get_candidates(reviews, meta, user_id)
    for product_id in recommendations:
        if product_id not in candidates:
            new_product_id = min(candidates, key=lambda x: distance(x, product_id))
            new_product_ids.append(new_product_id)
            non_existing_product_count += 1
        else:
            new_product_ids.append(product_id)

    return new_product_ids, non_existing_product_count