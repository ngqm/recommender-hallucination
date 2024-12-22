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
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
import nltk
from typing import List, Dict, Union, Tuple
import torch
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, BertForSequenceClassification, BertTokenizer

import warnings
from transformers import logging
logging.set_verbosity_error()
warnings.filterwarnings("ignore")


class HallucinationMetrics:
    def __init__(self):
        # Initialize NLTK components
        try:
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            nltk.download('punkt')
        except:
            print("NLTK data already downloaded")
            
        # Initialize metrics components
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoother = SmoothingFunction()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Store the bertscore model to avoid reloading
        self._bertscore_model = None
        self._bertscore_batch_size = 16

    def _initialize_bertscore_model(self):
        """Lazy initialization of BERTScore model"""
        if self._bertscore_model is None:
            # Initialize model here if needed
            pass

    def _clear_cuda_cache(self):

        if self.device == "cuda":
            torch.cuda.empty_cache()

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate ROUGE scores"""
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }

    def calculate_meteor(self, reference: str, candidate: str) -> float:
        """Calculate METEOR score"""
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)
        references = [reference_tokens]  # List of reference token lists
        return meteor_score(references, candidate_tokens)

    def calculate_bleu(self, reference: str, candidate: str) -> Dict[str, float]:
        """Calculate BLEU scores"""
        reference_tokens = [nltk.word_tokenize(reference)]
        candidate_tokens = nltk.word_tokenize(candidate)
        
        bleu_scores = {}
        for i in range(1, 5):
            bleu_scores[f'bleu{i}'] = sentence_bleu(
                reference_tokens,
                candidate_tokens,
                weights=tuple([1.0/i]*i),
                smoothing_function=self.smoother.method1
            )
        return bleu_scores

    def calculate_bertscore(self, references: List[str], candidates: List[str]) -> Dict[str, List[float]]:
        """Calculate BERTScore with memory management"""
        # Store original device setting
        original_device = self.device
        
        try:
            self._initialize_bertscore_model()
            
            # Process in smaller batches
            results = {
                'bertscore_precision': [],
                'bertscore_recall': [],
                'bertscore_f1': []
            }
            
            for i in range(0, len(references), self._bertscore_batch_size):
                try:
                    # Try to use original device (GPU if available)
                    self.device = original_device
                    
                    batch_refs = references[i:i + self._bertscore_batch_size]
                    batch_cands = candidates[i:i + self._bertscore_batch_size]
                    
                    P, R, F1 = score(
                        batch_refs, 
                        batch_cands, 
                        model_type="allenai/led-base-16384",
                        lang='en',
                        device=self.device,
                        batch_size=self._bertscore_batch_size
                    )
                    
                    results['bertscore_precision'].extend(P.tolist())
                    results['bertscore_recall'].extend(R.tolist())
                    results['bertscore_f1'].extend(F1.tolist())
                    
                    # Clear cache after each batch
                    self._clear_cuda_cache()
                    
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        print(f"WARNING: CUDA out of memory for batch {i//self._bertscore_batch_size + 1}. Using CPU for this batch.")
                        # Temporarily use CPU for this batch
                        self.device = "cpu"
                        
                        # Retry the same batch on CPU
                        P, R, F1 = score(
                            batch_refs, 
                            batch_cands, 
                            model_type="allenai/led-base-16384",
                            lang='en',
                            device=self.device,
                            batch_size=self._bertscore_batch_size
                        )
                        
                        results['bertscore_precision'].extend(P.tolist())
                        results['bertscore_recall'].extend(R.tolist())
                        results['bertscore_f1'].extend(F1.tolist())
                    else:
                        raise e
            
            # Restore original device setting
            self.device = original_device
            return results
            
        except Exception as e:
            print(f"Error in BERTScore calculation: {str(e)}")
            # Restore original device setting before raising the error
            self.device = original_device
            raise e
            
    def evaluate_all_metrics(self, reference: str, candidate: str, include_bertscore: bool = True) -> Dict[str, float]:
        """Calculate all metrics with memory management"""
        results = {}
        
        # Calculate ROUGE scores
        rouge_scores = self.calculate_rouge(reference, candidate)
        results.update(rouge_scores)
        
        # Calculate METEOR score
        meteor = self.calculate_meteor(reference, candidate)
        results['meteor'] = meteor
        
        # Calculate BLEU scores
        bleu_scores = self.calculate_bleu(reference, candidate)
        results.update(bleu_scores)
        
        # Calculate BERTScore if requested
        if include_bertscore:
            bertscore = self.calculate_bertscore([reference], [candidate])
            results['bertscore_precision'] = bertscore['bertscore_precision'][0]
            results['bertscore_recall'] = bertscore['bertscore_recall'][0]
            results['bertscore_f1'] = bertscore['bertscore_f1'][0]
            self._clear_cuda_cache()
        
        return results

    def evaluate_batch(self, references: List[str], candidates: List[str], include_bertscore: bool = True) -> pd.DataFrame:
        """Evaluate multiple reference-candidate pairs."""
        all_results = []
        
        for ref, cand in zip(references, candidates):
            results = self.evaluate_all_metrics(ref, cand, include_bertscore)
            all_results.append(results)
            
        return pd.DataFrame(all_results)



class IterativeUserProfiler:
    def __init__(self, df_review: pd.DataFrame, df_meta: pd.DataFrame, api_key: str):
        self.df_review = df_review
        self.df_meta = df_meta
        self.client = OpenAI(api_key=api_key)
        self.metrics_calculator = HallucinationMetrics()
    
    def get_user_interactions(self, user_id: str) -> pd.DataFrame:
        """Get user's interactions sorted chronologically"""
        user_data = pd.merge(
            self.df_review[self.df_review['user_id'] == user_id],
            self.df_meta,
            on='parent_asin',
            suffixes=('', '_meta')
        )
        
        # Sort by review time if available
        user_data['timestamp'] = pd.to_datetime(user_data['timestamp'], unit='ms')
        user_data = user_data.sort_values('timestamp')
        return user_data.reset_index(drop=True)
    
    def create_user_summary(self, user_data: pd.DataFrame) -> str:

        """Create summary for user interactions"""
        # Handle empty data
        if isinstance(user_data, pd.DataFrame) and user_data.empty:
            return "No purchase history found"
        
        # Convert Series to DataFrame if necessary
        if isinstance(user_data, pd.Series):
            user_data = pd.DataFrame([user_data])
        
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
    

    def create_update_prompt(self, current_profile: str, interaction_summary: str) -> str:
        """Creates a prompt for updating the user profile"""
        if current_profile:
            prompt = f"""Given the current user profile:

{current_profile}

The user has made a new interaction:
{interaction_summary}

Please update the summary of the user's preferences and behavior, incorporating this new information in under 256 words. Do not repeat specific product names or details."""
        else:
            prompt = f"""Based on this first interaction, please create an initial user summary by summarizing this person's genuine preferences and 
        patterns in under 256 words. Focus only on keywords and clear trends supported by multiple interactions. Do not repeat specific product names or details.

{interaction_summary}

Please analyze this first interaction and create an initial profile of the user's preferences."""

        return prompt
    
    def get_llm_update(self, prompt: str) -> str:
        """Get updated profile from LLM"""
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant maintaining user profiles. Update the profile based on new interactions while maintaining consistency with previous behavior patterns. The recent interaction is usually the most important"},
                    {"role": "user", "content": prompt}
                ]
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error with OpenAI API: {e}")
            return None
    
    def create_reference_summary(self, interactions: pd.DataFrame) -> str:

        summary = []
        
        # Process each interaction chronologically
        for idx, interaction in interactions.iterrows():
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
    

    def create_iterative_profile(self, user_id: str, output_dir: str = "iterative_profiles") -> Tuple[List[dict], dict]:
        """Creates and updates user profile iteratively with each interaction, excluding the last one as target"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Get chronologically sorted interactions
        interactions = self.get_user_interactions(user_id)
        if len(interactions) < 2:  # Need at least 2 interactions
            return [], None
        
        # Separate training interactions and target
        training_interactions = interactions.iloc[:-1]
        target_interaction = interactions.iloc[-1]
        
        profile_updates = []
        current_profile = ""
        current_interactions = pd.DataFrame()
        
        # Process each training interaction
        for idx, interaction in training_interactions.iterrows():
            # Add current interaction to accumulated interactions
            current_interactions = pd.concat([current_interactions, interaction.to_frame().T])
            
            # Create reference summary from all interactions up to this point
            reference_summary = self.create_reference_summary(current_interactions)
            
            # Create summary of current interaction only (for the prompt)
            interaction_summary = self.create_user_summary(interaction)
            
            # Create/update profile
            update_prompt = self.create_update_prompt(current_profile, interaction_summary)
            profile_text = self.get_llm_update(update_prompt)
            current_profile = profile_text
            
            # Calculate hallucination metrics
            metrics = self.metrics_calculator.evaluate_all_metrics(
                reference=reference_summary,
                candidate=profile_text
            )
            
            # Save update
            update = {
                'interaction_number': idx + 1,
                'interaction_summary': interaction_summary,
                'reference_summary': reference_summary,
                'profile': profile_text,
                'metrics': metrics
            }
            profile_updates.append(update)
            
            # Save to file
            with open(f"{output_dir}/user_{user_id}_update_{idx+1}.json", 'w') as f:
                json.dump(update, f, indent=2)
            
        
        # Save target interaction separately
        target_interaction_dict = target_interaction.to_dict()
        
        # Convert timestamp to string
        if 'timestamp' in target_interaction_dict:
            target_interaction_dict['timestamp'] = str(target_interaction_dict['timestamp'])

        target_data = {
            'interaction_number': len(interactions),
            'interaction_summary': self.create_user_summary(target_interaction),
            'reference_summary': self.create_reference_summary(interactions),
            'actual_interaction': target_interaction_dict
        }
        
        with open(f"{output_dir}/user_{user_id}_target.json", 'w') as f:
            json.dump(target_data, f, indent=2)
        
        # Create summary DataFrame of metrics across iterations
        metrics_df = pd.DataFrame([update['metrics'] for update in profile_updates])
        metrics_df['interaction_number'] = range(1, len(metrics_df) + 1)
        metrics_df.to_csv(f"{output_dir}/user_{user_id}_metrics.csv", index=False)
        
        return profile_updates, target_data
    
    def process_multiple_users(self, user_ids: List[str], output_dir: str = "iterative_profiles_b", batch_size: int = 10) -> Tuple[pd.DataFrame, Dict[str, dict]]:
        """Process users in batches to manage memory"""
        all_metrics = []
        all_targets = {}
        
        # Create main output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process users in batches
        for i in range(0, len(user_ids), batch_size):
            batch_users = user_ids[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}, users {i+1} to {min(i+batch_size, len(user_ids))}")
            
            for user_id in batch_users:
                try:
                    # Create user-specific directory
                    user_dir = os.path.join(output_dir, f"user_{user_id}")
                    os.makedirs(user_dir, exist_ok=True)
                    
                    # Process user
                    profile_updates, target_data = self.create_iterative_profile(user_id, output_dir=user_dir)
                    
                    if profile_updates and target_data:
                        # Extract metrics
                        for update in profile_updates:
                            metrics = update['metrics'].copy()
                            metrics['user_id'] = user_id
                            metrics['interaction_number'] = update['interaction_number']
                            all_metrics.append(metrics)
                        
                        all_targets[user_id] = target_data
                        
                except Exception as e:
                    print(f"Error processing user {user_id}: {e}")
                    continue
            
            # Clear CUDA cache after each batch
            torch.cuda.empty_cache()
        
        # Save results
        all_metrics_df = pd.DataFrame(all_metrics)
        all_metrics_df.to_csv(os.path.join(output_dir, "all_users_metrics.csv"), index=False)
        
        with open(os.path.join(output_dir, "all_targets.json"), 'w') as f:
            json.dump(all_targets, f, indent=2)
        
        return all_metrics_df, all_targets
    

if __name__ == "__main__":
        
    review_data_file = "data/All_Beauty_processed.jsonl"
    meta_data_file = "data/meta_All_Beauty_processed.jsonl"
    users_file = "data/All_Beauty_test_users.txt"

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

    iterative_profiler = IterativeUserProfiler(df_review, df_meta, OPENAI_API_KEY)
    test_users = eligible_users
    iterative_profiler.process_multiple_users(test_users)
