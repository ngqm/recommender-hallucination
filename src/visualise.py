# visualisation types:
# (1) hallucination metric against recommender metric, 
# (2) stage 1 metric against recommender metric,
# (3) hallucination metric against stage 1 metric, where
# recommender metrics = ["ndcg", "hr", "accuracy"]
# hallucination_metrics = ["llm", "non_existing_product_count"]
# stage_1_metrics = ["bleu (1,2,3,4)", "rouge (1,2,l)", "bert_score (precision, recall, f1)"]

# since predicting without candidate results in near-zero scores for most cases, we will only use candidate task for visualisation

import os
import sys

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("fivethirtyeight")
plt.rcParams['figure.facecolor'] = 'white'  # Figure background
plt.rcParams['axes.facecolor'] = 'white'   # Axes background
plt.rcParams['savefig.facecolor'] = 'white'  # Background of saved figures

from utils import get_data


class Visualise:

    def __init__(self, dataset, profile_type):

        self.dataset = dataset  # "All_Beauty" or "Movies_and_TV"
        self.profile_type = profile_type  # "iterative", "preference_negative", "preference_positive", "vanilla", "vanilla_structured", 

        self.shortname = {"All_Beauty": "beauty", "Movies_and_TV": "movies"}[dataset]

        self.reviews, self.meta, self.train_users, self.test_users = get_data(dataset)
        self.type_1_metrics = pd.read_csv(f"../type_1_eval/{self.shortname}/{self.profile_type}/metrics.csv")
        self.type_2_metrics = pd.read_csv(f"../type_2_eval/{self.shortname}/{self.profile_type}/metrics.csv")

    def _get_hallucination_data(self, hallucination_metric, which_type):
        """
        Get data for the given hallucination metric.

        :param hallucination_metric: str
        """

        if which_type == "type_1":
            metrics = self.type_1_metrics
        elif which_type == "type_2":
            metrics = self.type_2_metrics

        if hallucination_metric == "consistency":
            consistency = []
        elif hallucination_metric == "non_existing_product_count":
            non_existing_product_counts = []
        elif hallucination_metric == "bleu":
            bleu = {1: [], 2: [], 3: [], 4: []}
        elif hallucination_metric == "rouge":
            rouge = {1: [], 2: [], "L": []}
        elif hallucination_metric == "bert_score":
            bert_score = {"precision": [], "recall": [], "f1": []}
        elif hallucination_metric == "meteor":
            meteor = []

        for subset in range(1, 11):
            if hallucination_metric == "consistency":
                # get llm-as-a-judge score for each subset
                subset_hallucination_rates = []
                for user in self.test_users[(subset-1)*50:subset*50]:
                    with open(f"../eval/{self.shortname}/{self.profile_type}/user_{user}_candidate.txt", 'r') as f:
                        answers = f.read().split(", ")
                    # the rate is the number of "Yes" answers divided by the total number of answers
                    hallucination_rate = answers.count("Yes") / len(answers)
                    subset_hallucination_rates.append(hallucination_rate)
                hallucination_rates.append(subset_hallucination_rates)
                if subset == 10:
                    hallucination_rates = np.array(hallucination_rates)
                    return hallucination_rates
            elif hallucination_metric == "non_existing_product_count":
                # get non_existing_product_count for each subset
                with open(f"../eval/{self.shortname}/{self.profile_type}/non_existing_product_count_subset_{subset}.txt", 'r') as f:
                    non_existing_product_counts.append(int(f.read()))
                if subset == 10:
                    non_existing_product_counts = np.array(non_existing_product_counts)
                    return non_existing_product_counts
            elif hallucination_metric == "bleu":
                # get bleu-1, bleu-2, bleu-3, bleu-4 for each subset
                subset_bleu = {1: [], 2: [], 3: [], 4: []}
                for user in self.test_users[(subset-1)*50:subset*50]:
                    # get data from eval/shortname/profile_type/metrics.csv
                    for n in [1, 2, 3, 4]:
                        subset_bleu[n].append(self.metrics.loc[self.metrics["user_id"] == user, f"bleu{n}"].values[0])
                for n in [1, 2, 3, 4]:
                    subset_bleu[n] = np.array(subset_bleu[n])
                    bleu[n].append(subset_bleu[n])
                if subset == 10:
                    return bleu
            elif hallucination_metric == "rouge":
                # get rouge-1, rouge-2, rouge-l for each subset
                subset_rouge = {1: [], 2: [], "L": []}
                for user in self.test_users[(subset-1)*50:subset*50]:
                    # get data from eval/shortname/profile_type/metrics.csv
                    for n in [1, 2, "L"]:
                        subset_rouge[n].append(self.metrics.loc[self.metrics["user_id"] == user, f"rouge{n}_f"].values[0])
                for n in [1, 2, "L"]:
                    subset_rouge[n] = np.array(subset_rouge[n])
                    rouge[n].append(subset_rouge[n])
                if subset == 10:
                    return rouge
            elif hallucination_metric == "bert_score":
                # get precision, recall, f1 for each subset
                subset_bert_score = {"precision": [], "recall": [], "f1": []}
                for user in self.test_users[(subset-1)*50:subset*50]:
                    # get data from eval/shortname/profile_type/metrics.csv
                    for metric in ["precision", "recall", "f1"]:
                        subset_bert_score[metric].append(self.metrics.loc[self.metrics["user_id"] == user, "bertscore_" + metric].values[0])
                for metric in ["precision", "recall", "f1"]:
                    subset_bert_score[metric] = np.array(subset_bert_score[metric])
                    bert_score[metric].append(subset_bert_score[metric])
                if subset == 10:
                    return bert_score
            elif hallucination_metric == "meteor":
                # get meteor for each subset
                subset_meteor = []
                for user in self.test_users[(subset-1)*50:subset*50]:
                    # get data from eval/shortname/profile_type/metrics.csv
                    subset_meteor.append(self.metrics.loc[self.metrics["user_id"] == user, "meteor"].values[0])
                subset_meteor = np.array(subset_meteor)
                meteor.append(subset_meteor)
                if subset == 10:
                    meteor = np.array(meteor)
                    return meteor

    def _get_recommender_data(self, recommender_metric):
        """Get data for the given recommender metric.

        :param recommender_metric: str, "ndcg", "hr", or "accuracy"
        """

        if recommender_metric == "ndcg":
            ndcg = {"@1": [], "@5": [], "@10": []}
        elif recommender_metric == "hr":
            hr = {"@1": [], "@5": [], "@10": []}
        elif recommender_metric == "accuracy":
            accuracy = []

        for subset in range(1, 11):

            if recommender_metric == "ndcg":
                # get ncdg@1, ndcg@5, ndcg@10 for each subset
                for k in [1, 5, 10]:
                    with open(f"../eval/{self.shortname}/{self.profile_type}/ndcg_{k}_subset_{subset}.txt", 'r') as f:
                        ndcg[f"@{k}"].append(float(f.read().split(", ")[1]))
                if subset == 10:
                    return ndcg
            elif recommender_metric == "hr":
                # get hr@1, hr@5, hr@10 for each subset
                for k in [1, 5, 10]:
                    with open(f"../eval/{self.shortname}/{self.profile_type}/hit_rate_{k}_subset_{subset}.txt", 'r') as f:
                        hr[f"@{k}"].append(float(f.read().split(", ")[1]))
                if subset == 10:
                    return hr
            elif recommender_metric == "accuracy":
                # get accuracy for each subset
                with open(f"../eval/{self.shortname}/{self.profile_type}/accuracy_subset_{subset}.txt", 'r') as f:
                    # positive, negative, and overall accuracy
                    accuracy.append([float(acc) for acc in f.read().split(", ")])
                if subset == 10:
                    accuracy = np.array(accuracy)
                    return accuracy
                
    def _get_hallucination_to_plot(self, hallucination_metric, hallucination):
        """Helper function to get hallucination data in the right format to plot.
        """
        if hallucination_metric == "llm":
            hallucination_to_plot = hallucination.mean(axis=1)
        elif hallucination_metric == "non_existing_product_count":
            hallucination_to_plot = hallucination
        elif hallucination_metric == "bleu":
            hallucination_to_plot = {n: np.mean(hallucination[n], axis=1) for n in [1, 2, 3, 4]}
        elif hallucination_metric == "rouge":
            hallucination_to_plot = {n: np.mean(hallucination[n], axis=1) for n in [1, 2, "L"]}
        elif hallucination_metric == "bert_score":
            hallucination_to_plot = {metric: np.mean(hallucination[metric], axis=1) for metric in ["precision", "recall", "f1"]}
        elif hallucination_metric == "meteor":
            hallucination_to_plot = np.mean(hallucination, axis=1)

        return hallucination_to_plot

    def _get_data_for_visualisation(self, hallucination_metric, recommender_metric):
        """
        Get data for the given hallucination metric and recommender metric. Wrapper function for _get_hallucination_data and _get_recommender_data.

        :param hallucination_metric: str, "llm", "non_existing_product_count", "bleu", "rouge", "bert_score", "meteor"
        :param recommender_metric: str, "ndcg", "hr", or "accuracy"
        """

        to_return = []

        to_return.append(self._get_hallucination_data(hallucination_metric))
        to_return.append(self._get_recommender_data(recommender_metric))

        return to_return

    def make_visualisation(self, hallucination_metric, recommender_metric):

        # get data for visualisation
        hallucination, performance = self._get_data_for_visualisation(hallucination_metric, recommender_metric)

        # visualise
        if recommender_metric == "ndcg" or recommender_metric == "hr":
            y_min, y_max = min([min(performance[f"@{k}"]) for k in [1, 5, 10]]), max([max(performance[f"@{k}"]) for k in [1, 5, 10]])
        elif recommender_metric == "accuracy":
            y_min, y_max = np.min(performance), np.max(performance)

        hallucination_to_plot = self._get_hallucination_to_plot(hallucination_metric, hallucination)

        fig, ax = plt.subplots(1, 3, figsize=(8, 3))
        if hallucination_metric in ["bleu", "rouge", "bert_score"]:
            fig, ax = plt.subplots(1, 4, figsize=(10, 3))
        # fig.patch.set_facecolor('white')
        # ax.set_facecolor('white')

        xlabel = {
            "llm": "Inconsistent Product Rate",
            "non_existing_product_count": "Non-Existing Product Count",
            "bleu": "BLEU", "rouge": "ROUGE", "bert_score": "BERT Score", "meteor": "METEOR"   
        }[hallucination_metric]

        if recommender_metric == "ndcg" or recommender_metric == "hr":

            for i, k in enumerate([1, 5, 10]):
                if hallucination_metric in ["llm", "non_existing_product_count", "meteor"]:
                    ax[i].scatter(
                        hallucination_to_plot, performance[f"@{k}"], 
                        color=f"tab:{['blue', 'red', 'green'][i]}", alpha=0.7)
                    z = np.polyfit(hallucination_to_plot, performance[f"@{k}"], 1)
                    p = np.poly1d(z)
                    ax[i].plot(hallucination_to_plot, p(hallucination_to_plot), 
                            color=f"tab:{['blue', 'red', 'green'][i]}", linestyle="-", alpha=0.25, linewidth=5)
                elif hallucination_metric in ["bleu", "rouge", "bert_score"]:
                    for n in hallucination_to_plot.keys():
                        name_n = 3 if n == "L" else n
                        if n == "precision":
                            name_n = 1
                        elif n == "recall":
                            name_n = 2
                        elif n == "f1":
                            name_n = 3
                        ax[i].scatter(
                            hallucination_to_plot[n], performance[f"@{k}"], 
                            color=f"tab:{['blue', 'red', 'green', 'olive'][name_n-1]}", alpha=0.7, label=f"{n}")
                        z = np.polyfit(hallucination_to_plot[n], performance[f"@{k}"], 1)
                        p = np.poly1d(z)
                        ax[i].plot(hallucination_to_plot[n], p(hallucination_to_plot[n]), 
                                color=f"tab:{['blue', 'red', 'green', 'olive'][name_n-1]}", linestyle="-", alpha=0.25, linewidth=5)
                if i == 1:
                    ax[i].set_xlabel(xlabel)
                # change all spines to white
                for spine in ["top", "right", "left", "bottom"]:
                    ax[i].spines[spine].set_color('white')
                ax[i].set_facecolor('white')
                ax[i].set_ylabel(f"{recommender_metric.upper()}@{k}")
                ax[i].set_ylim(y_min-0.05, y_max+0.05)

        elif recommender_metric == "accuracy":

            for i, label in enumerate(["Positive", "Negative", "Overall"]):

                if hallucination_metric in ["llm", "non_existing_product_count", "meteor"]:
                    ax[i].scatter(
                        hallucination_to_plot, performance[:, i], 
                        color=f"tab:{['blue', 'red', 'green'][i]}", alpha=0.7)
                    z = np.polyfit(hallucination_to_plot, performance[:, i], 1)
                    p = np.poly1d(z)
                    ax[i].plot(hallucination_to_plot, p(hallucination_to_plot), 
                            color=f"tab:{['blue', 'red', 'green'][i]}", linestyle="-", alpha=0.25, linewidth=5)
                elif hallucination_metric in ["bleu", "rouge", "bert_score"]:
                    for n in hallucination_to_plot.keys():
                        name_n = 3 if n == "L" else n
                        if n == "precision":
                            name_n = 1
                        elif n == "recall":
                            name_n = 2
                        elif n == "f1":
                            name_n = 3
                        ax[i].scatter(
                            hallucination_to_plot[n], performance[:, i], 
                            color=f"tab:{['blue', 'red', 'green', 'olive'][name_n-1]}", alpha=0.7, label=f"{n}")
                        z = np.polyfit(hallucination_to_plot[n], performance[:, i], 1)
                        p = np.poly1d(z)
                        ax[i].plot(hallucination_to_plot[n], p(hallucination_to_plot[n]), 
                                color=f"tab:{['blue', 'red', 'green', 'olive'][name_n-1]}", linestyle="-", alpha=0.25, linewidth=5)
                        
                if i == 1:
                    ax[i].set_xlabel(xlabel)
                # change all spines to white
                for spine in ["top", "right", "left", "bottom"]:
                    ax[i].spines[spine].set_color('white')
                ax[i].set_ylabel(f"Acc. {label}")
                ax[i].set_ylim(y_min-0.05, y_max+0.05)

        if hallucination_metric in ["bleu", "rouge", "bert_score"]:
            for n in hallucination_to_plot.keys():
                name_n = 3 if n == "L" else n
                if n == "precision":
                    name_n = 1
                elif n == "recall":
                    name_n = 2
                elif n == "f1":
                    name_n = 3
                ax[i+1].scatter([], [], color=f"tab:{['blue', 'red', 'green', 'olive'][name_n-1]}", label=f"{n}")
            ax[i+1].axis("off")
            ax[i+1].legend()

        plt.suptitle(f"dataset: {self.shortname}, profile type: {self.profile_type}")
        plt.tight_layout()
        if os.path.isdir(f"../visualisation/{self.shortname}/{self.profile_type}") is False:
            os.makedirs(f"../visualisation/{self.shortname}/{self.profile_type}")
        plt.savefig(f"../visualisation/{self.shortname}/{self.profile_type}/{hallucination_metric}_{recommender_metric}.png", dpi=300)
        plt.close()

    def make_visualisation_internal(self, stage_1_metric, stage_2_metric):
        """Make visualisation to show the correlation between hallucination in stage 1 and hallucination in stage 2.

        :param stage_1_metric: str, "bleu", "rouge", "bert_score", "meteor"
        :param stage_2_metric: str, "llm", "non_existing_product_count"
        """

        label_dict = {
            "llm": "Inconsistency",
            "non_existing_product_count": "# $!\exists$ Products",
            "bleu": "BLEU", "rouge": "ROUGE", "bert_score": "BERT Score", "meteor": "METEOR"   
        }
        
        x_label = label_dict[stage_1_metric]
        y_label = label_dict[stage_2_metric]
        
        stage_1_data = self._get_hallucination_data(stage_1_metric)
        stage_2_data = self._get_hallucination_data(stage_2_metric)

        stage_1_to_plot = self._get_hallucination_to_plot(stage_1_metric, stage_1_data)
        stage_2_to_plot = self._get_hallucination_to_plot(stage_2_metric, stage_2_data)

        if stage_1_metric in ["bleu", "rouge", "bert_score"]:
            fig, ax = plt.subplots(1, 2, figsize=(5, 3))
            for n in stage_1_to_plot.keys():
                name_n = 3 if n == "L" else n
                if n == "precision":
                    name_n = 1
                elif n == "recall":
                    name_n = 2
                elif n == "f1":
                    name_n = 3
                ax[0].scatter(
                    stage_1_to_plot[n], stage_2_to_plot, 
                    color=f"tab:{['blue', 'red', 'green', 'olive'][name_n-1]}", alpha=0.7, label=f"{n}")
                z = np.polyfit(stage_1_to_plot[n], stage_2_to_plot, 1)
                p = np.poly1d(z)
                ax[0].plot(stage_1_to_plot[n], p(stage_1_to_plot[n]), 
                        color=f"tab:{['blue', 'red', 'green', 'olive'][name_n-1]}", linestyle="-", alpha=0.25, linewidth=5)
            
            ax[0].set_xlabel(x_label)
            ax[0].set_ylabel(y_label)

            for spine in ["top", "right", "left", "bottom"]:
                ax[0].spines[spine].set_color('white')
            ax[0].set_facecolor('white')

            # make legend in the second plot
            for n in stage_1_to_plot.keys():
                name_n = 3 if n == "L" else n
                if n == "precision":
                    name_n = 1
                elif n == "recall":
                    name_n = 2
                elif n == "f1":
                    name_n = 3
                ax[1].scatter([], [], color=f"tab:{['blue', 'red', 'green', 'olive'][name_n-1]}", label=f"{n}")
            ax[1].axis("off")
            ax[1].legend()
            plt.suptitle(f"dataset: {self.shortname}, profile type: {self.profile_type}")

        else:
            fig, ax = plt.subplots(1, 1, figsize=(3, 3))
            ax.scatter(stage_1_to_plot, stage_2_to_plot, color="tab:blue", alpha=0.7)
            z = np.polyfit(stage_1_to_plot, stage_2_to_plot, 1)
            p = np.poly1d(z)
            ax.plot(stage_1_to_plot, p(stage_1_to_plot), color="tab:blue", linestyle="-", alpha=0.25, linewidth=5)

            ax.set_xlabel(x_label)
            ax.set_ylabel(y_label)
            for spine in ["top", "right", "left", "bottom"]:
                ax.spines[spine].set_color('white')
            ax.set_facecolor('white')
        
            plt.suptitle(f"dataset: {self.shortname}\nprofile type: {self.profile_type}")
        plt.tight_layout()
        if os.path.isdir(f"../visualisation_internal/{self.shortname}/{self.profile_type}") is False:
            os.makedirs(f"../visualisation_internal/{self.shortname}/{self.profile_type}")
        plt.savefig(f"../visualisation_internal/{self.shortname}/{self.profile_type}/{stage_1_metric}_{stage_2_metric}.png", dpi=300)
        plt.close()

if __name__=="__main__":

    # call signature: python visualise.py dataset profile_type
    dataset = sys.argv[1]  # "All_Beauty" or "Movies_and_TV"
    profile_type = sys.argv[2]  # "vanilla", "vanilla_structured", "iterative"
    # hallucination_metric = sys.argv[3]  # "llm", "non_existing_product_count", "bleu", "rouge", "bert_score", "meteor"
    # recommender_metric = sys.argv[4]  # "ndcg", "hr", or "accuracy"
    stage_1_metric = sys.argv[3]  # "bleu", "rouge", "bert_score", "meteor"
    stage_2_metric = sys.argv[4]  # "llm", "non_existing_product_count"

    visualiser = Visualise(dataset, profile_type)
    # visualiser.make_visualisation(hallucination_metric, recommender_metric)
    visualiser.make_visualisation_internal(stage_1_metric, stage_2_metric)