import subprocess
from itertools import product

# List of commands to run in separate tmux sessions
datasets = ["All_Beauty", "Movies_and_TV"]
profile_types = ["iterative", "preference_positive", "preference_negative", "vanilla", "vanilla_structured"]
subsets = [str(i) for i in range(1, 11)]

# Loop through commands and create a tmux session for each
for dataset, profile_type, subset in product(datasets, profile_types, subsets):
    command = f"python recommend.py {dataset} {profile_type} {subset}"
    session_name = f"{dataset} {profile_type} {subset}"
    # Create a new tmux session in detached mode
    subprocess.run(["tmux", "new-session", "-d", "-s", session_name, command])

# List active tmux sessions for user convenience
subprocess.run(["tmux", "list-sessions"])
