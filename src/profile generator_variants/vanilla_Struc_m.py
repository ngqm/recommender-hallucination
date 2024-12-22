import os
import re
import time
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from openai_api import OPENAI_API_KEY
from collections import defaultdict
import nltk
from typing import List, Dict, Union, Tuple
import torch
import warnings
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")

class UserProfiler:
    def __init__(self, df_review: pd.DataFrame, df_meta: pd.DataFrame, api_key: str):
        self.df_review = df_review
        self.df_meta = df_meta
        self.client = OpenAI(api_key=api_key)
    
    def get_user_data(self, user_id: str) -> pd.DataFrame:
        """Get merged user purchase history data"""
        user_data = pd.merge(
            self.df_review[self.df_review['user_id'] == user_id],
            self.df_meta,
            on='parent_asin',
            suffixes=('', '_meta')
        )
        # Sort by review time if available
        user_data['timestamp'] = pd.to_datetime(user_data['timestamp'], unit='ms')
        user_data = user_data.sort_values('timestamp')
        
        return user_data

    def create_reference(self, user_data: pd.DataFrame) -> str:
        """
        Create reference for comparison with summary
        """
        if user_data.empty:
            return "No purchase history found"
        
        summary = []
        
        # Process each interaction chronologically
        for idx, interaction in user_data.iterrows():
            # Define fields to include in summary with their labels
            fields_to_check = [
                ('title_meta', 'Product Title'),
                ('store', 'Store'),
                ('categories', 'Categories'),
                ('description', 'Product Description'),
                ('details', 'Product Details'),
                ('text', 'Review Text'),
                ('title', 'Review Title')

            ]
            
            # Process each field
            for field, label in fields_to_check:
                value = interaction.get(field, None)  # Safely get field
                if value is None:
                    continue
                
                if isinstance(value, np.ndarray):
                    is_valid = not np.any(pd.isna(value))  # Check array validity
                elif isinstance(value, list):
                    is_valid = len(value) > 0 and not any(pd.isna(item) for item in value)  # Check list items
                else:
                    is_valid = not pd.isna(value)
                
                if is_valid:
                    if field == 'categories' and isinstance(value, list):
                        summary.append(f"{', '.join(value)}")
                    else:
                        summary.append(f"{value}")
        
        return "\n".join(summary)

    def create_user_summary(self, user_data: pd.DataFrame) -> str:
        """
        Create minimal summary using only explicitly stated data from user interactions
        
        Args:
            user_data (pd.DataFrame): DataFrame containing user purchase and review data
            
        Returns:
            str: Formatted summary of user interactions
        """
        if user_data.empty:
            return "No purchase history found"
        
        summary = []
        
        # Process each interaction chronologically
        for idx, interaction in user_data.iterrows():
            summary.append("=== Purchase Information ===")
            
            # Define fields to include in summary with their labels
            fields_to_check = [
                ('title_meta', 'Product Title'),
                ('timestamp', 'Review Date'),
                ('store', 'Store'),
                ('categories', 'Categories'),
                ('description', 'Product Description'),
                ('average_rating', 'Average Rating'),
                ('details', 'Product Details'),
                ('text', 'Review Text'),
                ('title', 'Review Title'),
                ('rating', 'Rating'),

            ]
            
            # Process each field
            for field, label in fields_to_check:
                value = interaction.get(field, None)  # Safely get field
                if value is None:
                    continue
                
                if isinstance(value, np.ndarray):
                    is_valid = not np.any(pd.isna(value))  # Check array validity
                elif isinstance(value, list):
                    is_valid = len(value) > 0 and not any(pd.isna(item) for item in value)  # Check list items
                else:
                    is_valid = not pd.isna(value)
                
                if is_valid:
                    if field == 'price':
                        summary.append(f"{label}: ${float(value):.2f}")
                    elif field == 'categories' and isinstance(value, list):
                        summary.append(f"{label}: {', '.join(value)}")
                    elif field == 'timestamp':
                        summary.append(f"{label}: {value.strftime('%Y/%m/%d')}")
                    else:
                        summary.append(f"{label}: {value}")
            
            summary.append("-" * 50)  # Separator between purchases
        
        return "\n".join(summary)

    
    #Prompt derived from Language-based uesr profiles for recommendation (Dai et al. 2024, WSDM)
    def create_llm_prompt(self, user_summary: str) -> str:
        prompt = f"""Based on this purchase history, analyze and summarize this person's genuine preferences and patterns in under 256 words. Think step by step:

    1. First, identify consistent patterns in ratings and sentiments
    2. Next, observe common themes across product purposes and use cases
    3. Then, note recurring quality preferences and price sensitivity patterns
    4. Finally, synthesize these observations into core preferences

    Remember to:
    - Focus only on keywords and  trends supported by multiple interactions
    - Avoid repeating specific product names or details
    - Base conclusions strictly on the provided purchase data

    Here's the purchase data:

    {user_summary}

    Structure your response as:
   
    SUMMARY:
    [Final synthesis of preferences and patterns]"""
        
        return prompt


    def get_llm_analysis(self, prompt: str) -> str:
        """Get analysis from OpenAI API"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant analyzing shopping behavior."},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return None

    def analyze_users(self, user_ids: list, summary_dir: str = "user_summaries") -> dict:
        """
        Analyze multiple users and save both prompts and responses for evaluation,
        excluding the last interaction as target
        """
        os.makedirs(summary_dir, exist_ok=True)
        analyses = {}
        
        for i, user_id in enumerate(user_ids):
            try:
                print(f"Processing user {i+1}/{len(user_ids)}: {user_id}")
                
                # Get user data
                user_data = self.get_user_data(user_id)
                
                if len(user_data) < 2:  # Need at least 2 interactions
                    print(f"Skipping user {user_id}: Not enough interactions")
                    continue
                
                # Separate training data and target
                training_data = user_data.iloc[:-1]
                target_interaction = user_data.iloc[-1]
                
                # Create summary excluding the target interaction
                user_summary = self.create_user_summary(training_data)
                reference_summary = self.create_reference(training_data)
                
                # Generate unified analysis###############
                prompt = self.create_llm_prompt(user_summary)
                #############################################
                analysis = self.get_llm_analysis(prompt)
                
                if analysis:
                    analyses[user_id] = analysis
                    
                    # Save all data
                    target_interaction_dict = target_interaction.to_dict()

                    # Convert timestamp to string
                    if 'timestamp' in target_interaction_dict:
                        target_interaction_dict['timestamp'] = str(target_interaction_dict['timestamp'])                    

                    # Save all data
                    user_data = {
                        "user_id": user_id,
                        "summary": user_summary,
                        "reference": reference_summary,
                        "prompt": prompt,
                        "raw_response": analysis,
                        "target_interaction": target_interaction_dict
                    }
                    
                    with open(f"{summary_dir}/user_{user_id}.json", 'w') as f:
                        json.dump(user_data, f, indent=2)
                
            except Exception as e:
                print(f"Error processing user {user_id}: {e}")
                continue
                
        return analyses

if __name__ == "__main__":
        
    review_data_file = "data/Movies_and_TV_processed.jsonl"
    meta_data_file = "data/meta_Movies_and_TV_processed.jsonl"
    users_file = "data/Movies_and_TV_test_users.txt"

    reviews = []
    with open(review_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                reviews.append(json.loads(line))
    df_review = pd.DataFrame(reviews)

    # Load meta data
    meta_items = []
    with open(meta_data_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                meta_items.append(json.loads(line))
    df_meta = pd.DataFrame(meta_items)

    eligible_users = []
    with open(users_file, 'r', encoding='utf-8') as f:
        for line in f:
            eligible_users.append(line.strip())

    profiler = UserProfiler(df_review, df_meta, OPENAI_API_KEY)

    test_users = eligible_users
    analyses = profiler.analyze_users(test_users, summary_dir="user_summaries_vanilla_Struc_m")
