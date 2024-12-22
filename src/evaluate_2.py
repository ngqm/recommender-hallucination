import os
import sys
import json
from tqdm import tqdm
from functools import reduce

import numpy as np
import pandas as pd

from utils import get_data, get_string_from_meta, get_profile, get_ground_truth, get_validated_products
from openai import OpenAI
from transformers import pipeline

from bert_score import score as bert_score


class Evaluate:

    def __init__(self, dataset, profile_type, subset):
        self.dataset = dataset  # "All_Beauty" or "Movies_and_TV"
        self.profile_type = profile_type  # "iterative", "vanilla", "vanilla_structured", "preference_positive", or "preference_negative"
        self.subset = subset  # from 1 to 10

        self.shortname = {"All_Beauty": "beauty", "Movies_and_TV": "movies"}[dataset]

        self.reviews, self.meta, self.train_users, self.test_users = get_data(dataset)

        if os.path.isdir(f"../type_2_eval/{self.shortname}/{self.profile_type}") is False:
            os.makedirs(f"../type_2_eval/{self.shortname}/{self.profile_type}")

    def _get_eval_prompt(self, product_ids, user_profile):
        """Produce a prompt for LLM-as-a-judge hallucination evaluation.

        :param product_ids: list[str], the parent_asin of the recommended products
        :param user_profile: str, the user profile in natural language

        :return: str, the prompt for hallucination evaluation
        """

        prompt = "###  User Profile: ###\n"
        prompt += user_profile
        prompt += "\n######\n"
        prompt += "Decide whether or not the following recommended products (10 in total) are supported by the user profile.\n"
        prompt += "Recommended Products:\n"
        prompt += "######\n"
        for product_id in product_ids:
            m = [m for m in self.meta if m['parent_asin'] == product_id][0]
            prompt += get_string_from_meta([m])[0]
            prompt += "\n######\n"
        prompt += "Answer with 'Yes' or 'No'.\n"
        prompt += "Do not explain. Just answer Yes or No.\n"
        prompt += "Separate each answer with a comma. There should be 10 answers in total.\n"

        return prompt
    
    def _get_gpt4_eval(self, product_id, user_profile):
        """Get hallucination evaluation result from GPT-4o-mini.

        :param product_id: str, the parent_asin of the recommended product
        :param user_profile: str, the user profile in natural language

        :return: str, the hallucination evaluation result
        """

        eval_prompt = self._get_eval_prompt(product_id, user_profile)
        with open('../openai_api', 'r') as f:
            api_key = f.read().strip()
        client = OpenAI(api_key=api_key)
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini-2024-07-18",
                messages=[
                    {"role": "user", "content": eval_prompt}
                ],
                max_tokens=300
            )
            hallucination = response.choices[0].message.content
            return hallucination
        except Exception as e:
            print(e)
            return None
    
    def get_gpt4_eval(self):
        """Get hallucination evaluations from GPT-4o-mini for the whole subset considered. Put all results in 'type_2_eval/' directory.
        """

        llm_df = {"user_id": [], "consistency": []}
        for user in tqdm(self.test_users[(subset-1)*50:subset*50]):
            user_profile = get_profile(self.shortname, self.profile_type, user)
            # candidate recommendation IDs
            with open(f"../recommendations/{self.shortname}/{self.profile_type}/user_{user}.json", 'r') as f:
                recommendations = json.load(f)["candidate"].split(", ")

            new_product_ids, _ = get_validated_products(self.reviews, self.meta, user, recommendations)
            eval = self._get_gpt4_eval(new_product_ids, user_profile)
            consistency = eval.split(", ").count("Yes")/10
            llm_df["user_id"].append(user)
            llm_df["consistency"].append(consistency)

        llm_df = pd.DataFrame(llm_df)
        llm_df.to_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/llm_subset_{subset}.csv", index=False)

    def _get_llama_eval(self, product_id, user_profile):
        """Get hallucination evaluation result from Llama-3.2-3B-Instruct.

        :param product_id: str, the parent_asin of the recommended product
        :param user_profile: str, the user profile in natural language
        
        :return: str, the hallucination evaluation result
        """

        eval_prompt = self._get_eval_prompt(product_id, user_profile)
        generator = pipeline(model="meta-llama/Llama-3.2-3B-Instruct", task="text-generation", device='cuda:0')
        try:
            response = generator([
                {"role": "user", "content": eval_prompt}
            ], max_new_tokens=300, pad_token_id=generator.tokenizer.eos_token_id)
            hallucination = response[0]['generated_text'][1]['content']
            return hallucination
        except Exception as e:
            print(e)
            return None
        
    def get_llama_eval(self):
        """Get hallucination evaluations from Llama-3.2-3B-Instruct for the whole subset considered. Put all results in 'type_2_eval/' directory.
        """

        llm_df = {"user_id": [], "consistency": []}
        for user in tqdm(self.test_users[(subset-1)*50:subset*50]):
            user_profile = get_profile(self.shortname, self.profile_type, user)
            # candidate recommendation IDs
            with open(f"../recommendations/{self.shortname}/{self.profile_type}/user_{user}.json", 'r') as f:
                recommendations = json.load(f)["candidate"].split(", ")

            new_product_ids, _ = get_validated_products(self.reviews, self.meta, user, recommendations)
            eval = self._get_llama_eval(new_product_ids, user_profile)
            consistency = eval.split(", ").count("Yes")/10
            llm_df["user_id"].append(user)
            llm_df["consistency"].append(consistency)

        llm_df = pd.DataFrame(llm_df)
        llm_df.to_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/llm_subset_{subset}.csv", index=False)

    def get_bert_score(self):
        """Get the BERTScore between user profiles and meta data of recommended products.
        """

        bert_score_df = {
            "bertscore_precision": [],
            "bertscore_recall": [],
            "bertscore_f1": [],
            "user_id": []}
        user_profiles = []
        meta_concats = []
        for user in tqdm(self.test_users[(subset-1)*50:subset*50]):
            bert_score_df["user_id"].append(user)
            user_profile = get_profile(self.shortname, self.profile_type, user)
            user_profiles.append(user_profile)

            # candidate recommendation IDs
            with open(f"../recommendations/{self.shortname}/{self.profile_type}/user_{user}.json", 'r') as f:
                recommendations = json.load(f)["candidate"].split(", ")

            new_product_ids, _ = get_validated_products(self.reviews, self.meta, user, recommendations)

            meta_concat = ""
            for product_id in new_product_ids:
                m = [m for m in self.meta if m['parent_asin'] == product_id][0]
                meta_concat += get_string_from_meta([m])[0]
            meta_concats.append(meta_concat)

        P, R, F1 = bert_score(user_profiles, meta_concats, lang='en')
        bert_score_df["bertscore_precision"] = P.tolist()
        bert_score_df["bertscore_recall"] = R.tolist()
        bert_score_df["bertscore_f1"] = F1.tolist()

        bert_score_df = pd.DataFrame(bert_score_df)
        bert_score_df.to_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/bert_score_subset_{subset}.csv", index=False)

    def get_nepc(self):
        """
        Get the number of non-existing products in generated recommendations.
        """
        
        nepc_df = {
            "user_id": [],
            "nepc": []
        }
        for user in tqdm(self.test_users[(subset-1)*50:subset*50]):
            nepc_df["user_id"].append(user)
            with open(f"../recommendations/{self.shortname}/{self.profile_type}/user_{user}.json", 'r') as f:
                recommendations = json.load(f)["candidate"].split(", ")
            _, nepc = get_validated_products(self.reviews, self.meta, user, recommendations)
            nepc_df["nepc"].append(nepc)

        nepc_df = pd.DataFrame(nepc_df)
        nepc_df.to_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/nepc_subset_{subset}.csv", index=False)

    def get_accuracy(self):
        """
        Get the accuracy of the binary task for the whole subset considered. Put the result in 'eval/' directory. The result consists of three numbers: the accuracy of positive task, the accuracy of negative task, and the overall accuracy.
        """

        accuracy_df = {
            "user_id": [],
            "true_positive": [],
            "true_negative": [],
            "accuracy": []
        }
        for user in tqdm(self.test_users[(subset-1)*50:subset*50]):
            accuracy_df["user_id"].append(user)

            with open(f"../recommendations/{self.shortname}/{self.profile_type}/user_{user}.json", 'r') as f:
                all_recommendations = json.load(f)
            # binary positive task
            binary_pos = all_recommendations["binary_pos"]
            # binary negative task
            binary_neg = all_recommendations["binary_neg"]
            accuracy_df["true_positive"].append(int(binary_pos == "Yes"))
            accuracy_df["true_negative"].append(int(binary_neg == "No"))
            accuracy_df["accuracy"].append(int((binary_pos == "Yes" and binary_neg == "No")))

        accuracy_df = pd.DataFrame(accuracy_df)
        accuracy_df.to_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/accuracy_subset_{subset}.csv", index=False)

    def get_hit_rate(self):
        """Get the hit rate @1, @5, and @10 of the top-at recommendation for the whole subset considered. Put the result in 'type_2_eval/' directory.
        """

        hit_rate_df = {
            "user_id": [],
            "hit_rate_at_1": [],
            "hit_rate_at_5": [],
            "hit_rate_at_10": []
        }
        for user in tqdm(self.test_users[(subset-1)*50:subset*50]):
            hit_rate_df["user_id"].append(user)
            ground_truth = get_ground_truth(self.shortname, user)
            # candidate recommendation IDs
            with open(f"../recommendations/{self.shortname}/{self.profile_type}/user_{user}.json", 'r') as f:
                recommendations = json.load(f)["candidate"].split(", ")
            new_product_ids, _ = get_validated_products(self.reviews, self.meta, user, recommendations)
            hit_rate_df["hit_rate_at_1"].append(int(ground_truth in new_product_ids[:1]))
            hit_rate_df["hit_rate_at_5"].append(int(ground_truth in new_product_ids[:5]))
            hit_rate_df["hit_rate_at_10"].append(int(ground_truth in new_product_ids[:10]))
        
        hit_rate_df = pd.DataFrame(hit_rate_df)
        hit_rate_df.to_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/hit_rate_subset_{subset}.csv", index=False)

    def get_ndcg(self):
        """Get the nDCG of the top-at recommendation for the whole subset considered. Put the result in 'type_2_eval/' directory. 
        
        :param at: int, the number of recommendations to consider. 10 at most.
        """

        ndcg_df = {
            "user_id": [],
            "ndcg_at_1": [],
            "ndcg_at_5": [],
            "ndcg_at_10": []
        }
        for user in tqdm(self.test_users[(subset-1)*50:subset*50]):
            ndcg_df["user_id"].append(user)
            ground_truth = get_ground_truth(self.shortname, user)
            # candidate recommendation IDs
            with open(f"../recommendations/{self.shortname}/{self.profile_type}/user_{user}.json", 'r') as f:
                recommendations = json.load(f)["candidate"].split(", ")
            new_product_ids, _ = get_validated_products(self.reviews, self.meta, user, recommendations)
            if ground_truth in new_product_ids[:1]:
                ndcg_df["ndcg_at_1"] = 1/np.log2(new_product_ids.index(ground_truth)+2)
            else:
                ndcg_df["ndcg_at_1"] = 0
            if ground_truth in new_product_ids[:5]:
                ndcg_df["ndcg_at_5"] = 1/np.log2(new_product_ids.index(ground_truth)+2)
            else:
                ndcg_df["ndcg_at_5"] = 0
            if ground_truth in new_product_ids[:10]:
                ndcg_df["ndcg_at_10"] = 1/np.log2(new_product_ids.index(ground_truth)+2)
            else:
                ndcg_df["ndcg_at_10"] = 0

        ndcg_df = pd.DataFrame(ndcg_df)
        ndcg_df.to_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/ndcg_subset_{subset}.csv", index=False)

    def combine_csv(self):
        """
        Combine all metrics into a csv that has 
            accuracy,
            bert_score,
            hit_rate@1, hit_rate@5, hit_rate@10,
            ndcg@1, ndcg@5, ndcg@10,
            non_existing_product_count,
            llm-as-a-judge hallucination evaluation
            user_id
        """
            
        accuracy_df = pd.read_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/accuracy_subset_{subset}.csv")
        bert_score_df = pd.read_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/bert_score_subset_{subset}.csv")
        hit_rate_df = pd.read_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/hit_rate_subset_{subset}.csv")
        ndcg_df = pd.read_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/ndcg_subset_{subset}.csv")
        llm_df = pd.read_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/llm_subset_{subset}.csv")
        nepc_df = pd.read_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/nepc_subset_{subset}.csv")

        data_frames = [accuracy_df, bert_score_df, hit_rate_df, ndcg_df, llm_df, nepc_df]
        
        # combine all metrics into one file with user_id as key
        df_merged = reduce(lambda left, right: pd.merge(left, right, on='user_id', how="outer"), data_frames)
        # save df_merged
        df_merged.to_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/metrics_subset_{subset}.csv", index=False)

if __name__ == "__main__":

    # call signature: python evaluate.py dataset profile_type subset
    dataset = sys.argv[1]  # "All_Beauty" or "Movies_and_TV"
    profile_type = sys.argv[2]  # "vanilla", "vanilla_structured", or "iterative"
    subset = int(sys.argv[3])  # from 1 to 10

    evaluator = Evaluate(dataset, profile_type, subset)

    print ("Getting llm-as-a-judge hallucination evaluations...")
    evaluator.get_gpt4_eval()
    print("Getting non-existing product count...")
    evaluator.get_nepc()
    print("Getting binary prediction accuracy...")
    evaluator.get_accuracy()

    print(f"Getting hit rate...")
    evaluator.get_hit_rate()
    print(f"Getting nDCG...")
    evaluator.get_ndcg()
    
    print("Getting BERTScore...")
    evaluator.get_bert_score()

    print("Combining all metrics...")
    evaluator.combine_csv()

    # once metrics are combined for all subsets, combine them into one file
    
    # if already exists, skip
    print("Combining all metrics for all subsets...")
    if os.path.isfile(f"../type_2_eval/{evaluator.shortname}/{evaluator.profile_type}/metrics.csv"):
        print("Metrics already combined for all subsets.")
        sys.exit()
    # if there is any missing subset, also skip
    for subset in range(1, 11):
        if os.path.isfile(f"../type_2_eval/{evaluator.shortname}/{evaluator.profile_type}/metrics_subset_{subset}.csv") is False:
            print(f"Missing subset {subset}. Cannot combine all metrics for all subsets yet.")
            sys.exit()
    for i in range(1, 11):
        with open(f"../type_2_eval/{evaluator.shortname}/{evaluator.profile_type}/metrics_subset_{i}.csv", 'r') as f:
            if i == 1:
                df = pd.read_csv(f)
            else:
                df = pd.concat([df, pd.read_csv(f)])
    df.to_csv(f"../type_2_eval/{evaluator.shortname}/{evaluator.profile_type}/metrics.csv", index=False)