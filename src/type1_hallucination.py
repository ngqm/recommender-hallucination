import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import nltk
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score
import torch
from transformers import logging

logging.set_verbosity_error()

class HallucinationMetrics:
    def __init__(self):
        try:
            nltk.download('wordnet', quiet=True)
            nltk.download('omw-1.4', quiet=True)
            nltk.download('punkt', quiet=True)
        except:
            pass
            
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoother = SmoothingFunction()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self._bertscore_batch_size = 16

    def calculate_rouge(self, reference: str, candidate: str) -> Dict[str, float]:
        scores = self.rouge_scorer.score(reference, candidate)
        return {
            'rouge1_f': scores['rouge1'].fmeasure,
            'rouge2_f': scores['rouge2'].fmeasure,
            'rougeL_f': scores['rougeL'].fmeasure
        }

    def calculate_meteor(self, reference: str, candidate: str) -> float:
        reference_tokens = nltk.word_tokenize(reference)
        candidate_tokens = nltk.word_tokenize(candidate)
        return meteor_score([reference_tokens], candidate_tokens)

    def calculate_bleu(self, reference: str, candidate: str) -> Dict[str, float]:
        reference_tokens = [nltk.word_tokenize(reference)]
        candidate_tokens = nltk.word_tokenize(candidate)
        
        return {f'bleu{i}': sentence_bleu(
            reference_tokens,
            candidate_tokens,
            weights=tuple([1.0/i]*i),
            smoothing_function=self.smoother.method1
        ) for i in range(1, 5)}

    def calculate_bertscore(self, references: List[str], candidates: List[str]) -> Dict[str, List[float]]:
        original_device = self.device
        results = {'bertscore_precision': [], 'bertscore_recall': [], 'bertscore_f1': []}
        
        try:
            for i in range(0, len(references), self._bertscore_batch_size):
                batch_refs = references[i:i + self._bertscore_batch_size]
                batch_cands = candidates[i:i + self._bertscore_batch_size]
                
                try:
                    self.device = original_device
                    P, R, F1 = score(
                        batch_refs, batch_cands,
                        model_type="allenai/led-base-16384",
                        lang='en',
                        device=self.device,
                        batch_size=self._bertscore_batch_size
                    )
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        self.device = "cpu"
                        P, R, F1 = score(
                            batch_refs, batch_cands,
                            model_type="allenai/led-base-16384",
                            lang='en',
                            device=self.device,
                            batch_size=self._bertscore_batch_size
                        )
                
                results['bertscore_precision'].extend(P.tolist())
                results['bertscore_recall'].extend(R.tolist())
                results['bertscore_f1'].extend(F1.tolist())
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            return results
        finally:
            self.device = original_device

    def evaluate_all_metrics(self, reference: str, candidate: str, include_bertscore: bool = True) -> Dict[str, float]:
        try:
            results = {}
            results.update(self.calculate_rouge(reference, candidate))
            results['meteor'] = self.calculate_meteor(reference, candidate)
            results.update(self.calculate_bleu(reference, candidate))
            
            if include_bertscore:
                bertscore = self.calculate_bertscore([reference], [candidate])
                results.update({
                    'bertscore_precision': bertscore['bertscore_precision'][0],
                    'bertscore_recall': bertscore['bertscore_recall'][0],
                    'bertscore_f1': bertscore['bertscore_f1'][0]
                })
            
            return results
        except Exception as e:
            print(f"Error calculating metrics: {str(e)}")
            return results

def _parse_summaries(response: str) -> dict:
    summaries = {"positive": "", "negative": ""}
    try:
        sections = response.split("SUMMARY:")
        if len(sections) >= 4:
            summaries["positive"] = sections[1].strip().split("\n\n")[0].strip()
            summaries["negative"] = sections[2].strip().split("\n\n")[0].strip()
    except Exception as e:
        print(f"Error parsing summaries: {e}")
    return summaries

def create_metric_plots(metrics_df: pd.DataFrame, output_dir: str, domain: str, profile_type: str):
    key_metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'meteor', 'bertscore_f1', 'bertscore_precision']
    metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'BERTScore F1', 'BERTScore Precision']
    
    plt.figure(figsize=(20, 12))
    for i, (metric, metric_name) in enumerate(zip(key_metrics, metric_names)):
        plt.subplot(3, 2, i+1)
        grouped_stats = metrics_df.groupby('num_interactions')[metric].agg(['mean', 'std']).reset_index()
        
        plt.bar(
            grouped_stats['num_interactions'],
            grouped_stats['mean'],
            yerr=grouped_stats['std'],
            capsize=5,
            alpha=0.7,
            color='skyblue',
            ecolor='black'
        )
        
        plt.title(f'{metric_name} by Interactions\n({domain} - {profile_type})')
        plt.xlabel('Number of Interactions')
        plt.ylabel('Score')
        plt.grid(True, axis='y', linestyle='--', alpha=0.7)
        plt.xticks(grouped_stats['num_interactions'])
        
        for _, row in grouped_stats.iterrows():
            plt.text(
                row['num_interactions'],
                row['mean'] + (row['std'] if not pd.isna(row['std']) else 0),
                f'{row["mean"]:.3f}',
                ha='center',
                va='bottom'
            )
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'metrics_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()

def analyze_summaries(input_dir: str, output_base_dir: str, domain: str, profile_type: str) -> pd.DataFrame:
    metrics_calculator = HallucinationMetrics()
    all_metrics = []
    output_dir = os.path.join(output_base_dir, domain, profile_type)
    os.makedirs(output_dir, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"Processing {len(files)} users in {input_dir}")
    
    for file in files:
        user_id = file.replace('user_', '').replace('.json', '')
        try:
            with open(os.path.join(input_dir, file), 'r') as f:
                user_data = json.load(f)
                metrics = metrics_calculator.evaluate_all_metrics(
                    reference=user_data['reference'],
                    candidate=user_data['raw_response']
                )
                metrics['user_id'] = user_id
                metrics['num_interactions'] = user_data['summary'].count('=== Purchase Information ===')
                all_metrics.append(metrics)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    create_metric_plots(metrics_df, output_dir, domain, profile_type)
    return metrics_df

def analyze_preference_summaries(input_dir: str, output_base_dir: str, domain: str) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    metrics_calculator = HallucinationMetrics()
    all_metrics_positive, all_metrics_negative = [], []
    
    output_dir_positive = os.path.join(output_base_dir, domain, 'preference_positive')
    output_dir_negative = os.path.join(output_base_dir, domain, 'preference_negative')
    os.makedirs(output_dir_positive, exist_ok=True)
    os.makedirs(output_dir_negative, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    print(f"Processing {len(files)} users in {input_dir}")
    
    for file in files:
        user_id = file.replace('user_', '').replace('.json', '')
        try:
            with open(os.path.join(input_dir, file), 'r') as f:
                user_data = json.load(f)
                summaries = _parse_summaries(user_data['raw_response'])
                num_interactions = user_data['summary'].count('=== Purchase Information ===')
                
                for summary_type, metrics_list in [
                    ('positive', all_metrics_positive),
                    ('negative', all_metrics_negative)
                ]:
                    if summaries[summary_type]:
                        metrics = metrics_calculator.evaluate_all_metrics(
                            reference=user_data['reference'],
                            candidate=summaries[summary_type]
                        )
                        metrics['user_id'] = user_id
                        metrics['num_interactions'] = num_interactions
                        metrics_list.append(metrics)
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
            continue
    
    results = []
    for metrics_list, output_dir, type_name in [
        (all_metrics_positive, output_dir_positive, 'preference_positive'),
        (all_metrics_negative, output_dir_negative, 'preference_negative')
    ]:
        if metrics_list:
            metrics_df = pd.DataFrame(metrics_list)
            metrics_df.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
            create_metric_plots(metrics_df, output_dir, domain, type_name)
            results.append(metrics_df)
        else:
            results.append(None)
    
    return tuple(results)

def main():
    domains = ['beauty', 'movies']
    profile_types = ['vanilla', 'vanilla_structured', 'preference']
    input_dir_mapping = {
        ('beauty', 'vanilla'): 'user_summaries_vanilla_b',
        ('beauty', 'vanilla_structured'): 'user_summaries_vanilla_Struc_b',
        ('beauty', 'preference'): 'user_summaries_pref_b',
        ('movies', 'vanilla'): 'user_summaries_vanilla_m',
        ('movies', 'vanilla_structured'): 'user_summaries_vanilla_Struc_m',
        ('movies', 'preference'): 'user_summaries_pref_m',
    }
    
    output_base_dir = 'type_1_eval'
    
    for domain in domains:
        for profile_type in profile_types:
            input_dir = input_dir_mapping.get((domain, profile_type))
            if input_dir and os.path.exists(input_dir):
                print(f"\nProcessing {domain} - {profile_type}")
                try:
                    if profile_type == 'preference':
                        analyze_preference_summaries(input_dir, output_base_dir, domain)
                    else:
                        analyze_summaries(input_dir, output_base_dir, domain, profile_type)
                    print(f"Completed analysis for {domain} - {profile_type}")
                except Exception as e:
                    print(f"Error processing {domain} - {profile_type}: {str(e)}")

if __name__ == "__main__":
    main()