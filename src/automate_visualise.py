import subprocess
from itertools import product

# List of commands to run in separate tmux sessions
datasets = ["All_Beauty", "Movies_and_TV"]
profile_types = ["iterative", "preference_positive", "preference_negative", "vanilla", "vanilla_structured"]
stage_1_metrics = ["bleu", "rouge", "bert_score", "meteor"]
stage_2_metrics = ["llm", "non_existing_product_count", "bert_score"]
hallucination_metrics = stage_1_metrics + stage_2_metrics
recommender_metrics = ["ndcg", "hr", "accuracy"]

# Loop through commands and create a tmux session for each
for dataset, profile_type, hallucination_metric, recommender_metric in product(datasets, profile_types, hallucination_metrics, recommender_metrics):
    command = f"python visualise.py {dataset} {profile_type} {hallucination_metric} {recommender_metric}"
    session_name = f"{dataset} {profile_type} {hallucination_metric} {recommender_metric}"
    # Create a new tmux session in detached mode
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, command])

# List active tmux sessions for user convenience
subprocess.run(["tmux", "list-sessions"])

for dataset, profile_types, stage_1_metric, stage_2_metric in product(datasets, profile_types, stage_1_metrics, stage_2_metrics):
    command = f"python visualise.py {dataset} {profile_types} {stage_1_metric} {stage_2_metric}"
    session_name = f"{dataset} {profile_types} {stage_1_metric} {stage_2_metric}"
    # Create a new tmux session in detached mode
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, command])

# List active tmux sessions for user convenience
subprocess.run(["tmux", "list-sessions"])
