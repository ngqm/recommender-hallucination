import os
import json
import pandas as pd
import warnings
import torch
import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple
from type1_hallucination import HallucinationMetrics

warnings.filterwarnings("ignore", category=UserWarning, module="torch._utils")

def create_comparison_plots(ind_means: pd.DataFrame, ind_stds: pd.DataFrame, 
                          agg_means: pd.DataFrame, agg_stds: pd.DataFrame,
                          output_dir: str):
    key_metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'meteor', 'bertscore_f1', 'bertscore_precision']
    metric_names = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'METEOR', 'BERTScore F1', 'BERTScore Precision']
    
    plt.figure(figsize=(20, 12))
    for i, (metric, metric_name) in enumerate(zip(key_metrics, metric_names)):
        plt.subplot(3, 2, i+1)
        if metric not in ind_means.columns or metric not in agg_means.columns:
            continue
            
        x = np.array(ind_means.index)
        plt.bar(x, ind_means[metric], 
                alpha=0.6, color='skyblue', yerr=ind_stds[metric],
                capsize=5, label='Individual', width=0.4)
        
        plt.plot(x, agg_means[metric], 
                color='red', linewidth=2, label='Aggregate Trend',
                marker='o', zorder=5)
        
        plt.fill_between(x, 
                        agg_means[metric] - agg_stds[metric],
                        agg_means[metric] + agg_stds[metric],
                        alpha=0.2, color='red')
        
        plt.title(f'{metric_name} Progression')
        plt.xlabel('Interaction Number')
        plt.ylabel('Score')
        plt.legend()
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.xticks(x)
        
        for x_val, y_val, std_val in zip(x, ind_means[metric], ind_stds[metric]):
            if not np.isnan(y_val):
                plt.text(x_val, 
                        y_val + (std_val if not np.isnan(std_val) else 0),
                        f'{y_val:.3f}',
                        ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, 'iterative_metrics_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

def analyze_iterative_profiles(base_dir: str, output_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    user_dirs = [d for d in os.listdir(base_dir) if d.startswith('user_')]
    print(f"Found {len(user_dirs)} users")
    
    metrics_calculator = HallucinationMetrics()
    all_individual_metrics = []
    all_aggregate_metrics = []
    
    for user_dir in user_dirs:
        user_id = user_dir.split('user_')[1]
        print(f"\nProcessing user: {user_id}")
        
        user_path = os.path.join(base_dir, user_dir)
        update_files = sorted([f for f in os.listdir(user_path) 
                             if f.startswith(f'user_{user_id}_update_')],
                             key=lambda x: int(x.split('_')[-1].split('.')[0]))
        
        for file in update_files:
            try:
                with open(os.path.join(user_path, file), 'r') as f:
                    update_data = json.load(f)
                
                individual_score = metrics_calculator.evaluate_all_metrics(
                    reference=update_data['interaction_summary'],
                    candidate=update_data['profile']
                )
                individual_score.update({
                    'interaction_number': update_data['interaction_number'],
                    'user_id': user_id
                })
                all_individual_metrics.append(individual_score)
                
                aggregate_score = update_data['metrics'].copy()
                aggregate_score.update({
                    'interaction_number': update_data['interaction_number'],
                    'user_id': user_id
                })
                all_aggregate_metrics.append(aggregate_score)
                
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                continue
    
    individual_df = pd.DataFrame(all_individual_metrics)
    aggregate_df = pd.DataFrame(all_aggregate_metrics)
    os.makedirs(output_dir, exist_ok=True)
    
    key_metrics = ['rouge1_f', 'rouge2_f', 'rougeL_f', 'meteor', 'bertscore_f1', 'bertscore_precision']
    ind_means = individual_df.groupby('interaction_number')[key_metrics].mean()
    ind_stds = individual_df.groupby('interaction_number')[key_metrics].std()
    agg_means = aggregate_df.groupby('interaction_number')[key_metrics].mean()
    agg_stds = aggregate_df.groupby('interaction_number')[key_metrics].std()
    
    create_comparison_plots(ind_means, ind_stds, agg_means, agg_stds, output_dir)
    
    individual_df.to_csv(os.path.join(output_dir, 'individual_metrics.csv'), index=False)
    aggregate_df.to_csv(os.path.join(output_dir, 'aggregate_metrics.csv'), index=False)
    pd.DataFrame({
        'Individual_Mean': individual_df[key_metrics].mean(),
        'Individual_Std': individual_df[key_metrics].std(),
        'Aggregate_Mean': aggregate_df[key_metrics].mean(),
        'Aggregate_Std': aggregate_df[key_metrics].std()
    }).to_csv(os.path.join(output_dir, 'summary_statistics.csv'))
    
    aggregate_sorted = aggregate_df.sort_values(by=['user_id', 'interaction_number'])
    final_metrics = aggregate_sorted.groupby('user_id').tail(1).copy()
    user_counts = aggregate_df.groupby('user_id')['interaction_number'].max().rename('num_interactions')
    final_metrics = final_metrics.merge(user_counts, on='user_id', how='left').drop('interaction_number', axis = 1)

    columns_order = [
        'rouge1_f', 'rouge2_f', 'rougeL_f', 'meteor', 
        'bleu1', 'bleu2', 'bleu3', 'bleu4', 
        'bertscore_precision', 'bertscore_recall', 'bertscore_f1', 
        'user_id', 'num_interactions'
    ]

    final_metrics[columns_order].to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)

    return individual_df, aggregate_df

def main():
    domains = ['beauty', 'movies']
    base_dirs = {
        'beauty': 'iterative_profiles_b',
        'movies': 'iterative_profiles_m'
    }
    
    for domain in domains:
        base_dir = base_dirs.get(domain)
        if base_dir and os.path.exists(base_dir):
            print(f"\nProcessing {domain} domain")
            output_dir = os.path.join('type_1_eval', domain, 'iterative')
            try:
                individual_df, aggregate_df = analyze_iterative_profiles(base_dir, output_dir)
                print(f"Completed analysis for {domain}")
            except Exception as e:
                print(f"Error processing {domain}: {str(e)}")
        else:
            print(f"Skipping {domain}: Directory not found")

if __name__ == "__main__":
    main()