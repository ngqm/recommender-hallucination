import pandas as pd
from itertools import product
import os
from scipy.stats import pearsonr

from utils import get_data

datasets = ["All_Beauty", "Movies_and_TV"]
shorthands = {"All_Beauty": "beauty", "Movies_and_TV": "movies"}
profile_types = ["iterative", "preference_positive", "preference_negative", "vanilla", "vanilla_structured"]
subsets = [str(i) for i in range(1, 11)]

for dataset, profile_type in product(datasets, profile_types):
    
    shorthand = shorthands[dataset]

    # get data
    reviews, meta, train_users, test_users = get_data(dataset)

    # compute correlation
    type_1_results= pd.read_csv(f"../type_1_eval/{shorthand}/{profile_type}/metrics.csv")
    type_2_results = pd.read_csv(f"../type_2_eval/{shorthand}/{profile_type}/metrics.csv")

    type_1_metrics = list(set(type_1_results.columns.to_list()) - {"user_id", "num_interactions"})
    type_2_metrics = ["bertscore_precision", "bertscore_recall", "bertscore_f1", "consistency", "nepc"]
    recommendation_metrics = [f"hit_rate_at_{k}" for k in [1, 5, 10]] + [f"ndcg_at_{k}" for k in [1, 5, 10]] + ["true_positive", "true_negative", "accuracy"]

    # create directory if not exists
    if not os.path.exists(f"../correlation/{shorthand}/{profile_type}"):
        os.makedirs(f"../correlation/{shorthand}/{profile_type}")

    file_names = ["type_1_type_2", "type_1_recommendation", "type_2_recommendation"]
    metrics_combos = [(type_1_metrics, type_2_metrics), (type_1_metrics, recommendation_metrics), (type_2_metrics, recommendation_metrics)]
    results_combos = [(type_1_results, type_2_results), (type_1_results, type_2_results), (type_2_results, type_2_results)]

    for metrics_combo, results_combo, file_name in zip(metrics_combos, results_combos, file_names):
        
        df = {
            "dataset": [],
            "profile_type": [],
            "metric_a": [],
            "metric_b": [],
            "correlation": [],
            "pvalue": [],
            "confidence_interval": []
        }
        metrics_a, metrics_b = metrics_combo
        results_a, results_b = results_combo

        for metric_a, metric_b in product(metrics_a, metrics_b):
            Xa, Xb = [], []
            for subset in subsets:
                user_subset = test_users[(int(subset) - 1) * 50: int(subset) * 50]
                type_1_metric_mean = results_a[results_a["user_id"].isin(user_subset)][metric_a].mean()
                type_2_metric_mean = results_b[results_b["user_id"].isin(user_subset)][metric_b].mean()
                Xa.append(type_1_metric_mean)
                Xb.append(type_2_metric_mean)
            stats_result = pearsonr(Xa, Xb)
            correlation = stats_result.statistic
            # treat nan
            if pd.isna(correlation):
                correlation = 0
                pvalue = 1
            else:
                pvalue = stats_result.pvalue
            ci = stats_result.confidence_interval(0.95)
            df["dataset"].append(dataset)
            df["profile_type"].append(profile_type)
            df["metric_a"].append(metric_a)
            df["metric_b"].append(metric_b)
            df["correlation"].append(correlation)
            df["pvalue"].append(pvalue)
            df["confidence_interval"].append((float(ci.low), float(ci.high)))
        df = pd.DataFrame(df)
        df.to_csv(f"../correlation/{shorthand}/{profile_type}/{file_name}.csv", index=False)
        