import os
import sys

import json
from tqdm import tqdm
import re

# import nltk
# from nltk.translate.bleu_score import sentence_bleu
# # if punkt is not downloaded, download it
# try:
#     nltk.data.find('tokenizers/punkt')
# except:
#     nltk.download('punkt')

# from transformers import pipeline
from openai import OpenAI

from utils import get_data, get_string_from_meta, get_candidates, get_profile


def get_recommendation_prompt(reviews, meta, user_id, user_profile, task, dataset):
    """Get the prompt for the recommendation task.

    :param list reviews: list of reviews
    :param list meta: list of metadata of products
    :param str user_id: user id
    :param str user_profile: user profile
    :param str task: task, either "candidate", "binary_pos", or "binary_neg"
    :param str dataset: dataset name, either "All_Beauty" or "Movies_and_TV"

    :return: prompt for the recommendation task.
    If task is "candidate", choose the best item from the following candidates to recommend for the user.
    If task is "binary", predict whether the user would like the new item or not. Two subtasks are available: "binary_pos" uses an item that will be interacted with in the data. "binary_neg" uses an item that will not be interacted with.

    :rtype: str
    """
    prompt = "###  User Profile: ###\n"
    prompt += user_profile
    prompt += "\n######\n"
    candidates = get_candidates(reviews, meta, user_id)
    if task == "candidate":
        prompt += "Choose the best 10 item from the following candidates to recommend for the user, ordered by relevance.\n"
        prompt += "Candidates:\n"
        prompt += "\n"
        for c in candidates:
            prompt += f"ID: {c}\n"
            m = [m for m in meta if m['parent_asin'] == c][0]
            prompt += get_string_from_meta([m])[0]
            prompt += "\n######\n"
        prompt += "Give the answer as a list of 10 IDs, separated by commas, among the given IDs.\n"
        prompt += "Do not explain. Just give the IDs.\n"
        return prompt
    prompt += "Predict whether the user would like the following new item or not.\n"
    prompt += "New Item:\n"
    if task == "binary_pos":
        new_item = candidates[-1]
    else:
        new_item = candidates[0]
    m = [m for m in meta if m['parent_asin'] == new_item][0]
    prompt += get_string_from_meta([m])[0]
    prompt += "\n######\n"
    prompt += "Answer with 'Yes' or 'No'.\n"
    prompt += "Do not explain. Just answer Yes or No.\n"
    return prompt


def generate_recommendation(
        reviews, meta, user_id, user_profile, 
        model_name, task, 
        dataset,
        generator=None, max_tokens=1024):
    """
    Generate a recommendation for the user with user_id based on the user profile and task.

    :param reviews: list of reviews
    :param meta: list of metadata of products
    :param user_id: user id
    :param user_profile: user profile
    :param model_name: model to use for generation, either "gpt-4o-mini-2024-07-18" or 
    "meta-llama/Llama-3.2-3B-Instruct"
    :param task: task, either "direct", "candidate", "binary_pos", or "binary_neg"
    :param dataset: dataset name, either "All_Beauty" or "Movies_and_TV"
    :param generator: generator object for model "meta-llama/Llama-3.2-3B-Instruct"
    :param max_tokens: maximum number of tokens to generate

    :return: generated recommendation
    :rtype: str
    """

    prompt = get_recommendation_prompt(reviews, meta, user_id, user_profile, task, dataset)

    if model_name.startswith("gpt"):
        with open('../openai_api', 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            print(e)
            return None
    else:
        response = generator([
            {"role": "user", "content": prompt}
        ], max_new_tokens=max_tokens, pad_token_id=generator.tokenizer.eos_token_id)
        return response[0]['generated_text'][1]['content']


if __name__=="__main__":

    # call signature: python recommend.py dataset profile_type subset
    dataset = sys.argv[1]  # "All_Beauty" or "Movies_and_TV"
    profile_type = sys.argv[2]  # "iterative", "preference_negative", "preference_positive", "vanilla", "vanilla_structured",
    subset = int(sys.argv[3])  # from 1 to 10

    shortname = {"All_Beauty": "beauty", "Movies_and_TV": "movies"}[dataset]

    reviews, meta, train_users, test_users = get_data(dataset)

    if os.path.isdir(f"../recommendations/{shortname}/{profile_type}") is False:
        os.makedirs(f"../recommendations/{shortname}/{profile_type}")
    
    for user_id in tqdm(test_users[(subset-1)*50:subset*50]):
        user_profile = get_profile(shortname, profile_type, user_id)
        results = {}
        for task in ["candidate", "binary_pos", "binary_neg"]:
            recommendation = generate_recommendation(
                reviews=reviews, meta=meta, 
                user_id=user_id, user_profile=user_profile, 
                model_name="gpt-4o-mini-2024-07-18",
                dataset=dataset,
                task=task)
            results[task] = recommendation
        with open(f"../recommendations/{shortname}/{profile_type}/user_{user_id}.json", 'w') as f:
            json.dump(results, f)