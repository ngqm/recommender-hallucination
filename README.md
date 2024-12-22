# An Investigation of Hallucination in LLM-Based Recommender Systems

The steps to reproduce the experiments are as follows:

## Prerequisites

1. Install required libraries with the following commands:
```
pip install matplotlib
pip install numpy
pip install openai
pip install transformers
```
2. Put your OpenAI API key into an `openai_api` file in the root directory.

## Data Preprocessing

First, download reviews and metadata of "All_Beauty" and "Movies_and_TV" categories 
from the Amazon Dataset at https://amazon-reviews-2023.github.io/. 
Place the downloaded files in the `data` directory:
```
data/
    All_Beauty.jsonl
    meta_All_Beauty.jsonl
    
    Movies_and_TV.jsonl
    meta_Movies_and_TV.jsonl
```

Run the following command to preprocess the data:

```bash
python preprocess.py
```

This command will produce the following files. For your convenience, we already ran the preprocessing step and 
provided the processed files in the `data` directory.

```
data/
    All_Beauty_processed.jsonl
    meta_All_Beauty_processed.jsonl
    All_Beauty_test_users.txt

    Movies_and_TV_processed.jsonl
    meta_Movies_and_TV_processed.jsonl
    Movies_and_TV_test_users.txt
```

## Profile Creation
To generate user profiles in the directory structure shown below, run `python generate_profile.py`

This step produces profiles in the following directories. For your convenience, we already ran the profile creation step and provided the profiles in the `profiles` directory.
```
profiles/
    beauty/
        iterative/
            user_{id}.json
            ...
        preference_negative/
            ...
        preference_positive/
            ...
        vanilla/
            ...
        vanilla_structured/ 
            ...
    movies/
        iterative/
            ...
        preference_negative/
            ...
        preference_positive/
            ...
        vanilla/
            ...
        vanilla_structured/ 
            ...
```

## Preliminary Experiments on Profile Types
We provide a preliminary experiment comparing aggregate and individual hallucinations in the iterative scheme. To run this analysis on the existing profiles, use `python iterative_type1_hallucination_analysis.py`
This script evaluates how the iterative profile scheme performs at each interaction step (comparing each step’s profile against user data). The input and output file structures are assumed to be the same as the profile directories illustrated above.
## Type 1 Hallucination Measurement
To compute Type 1 hallucination metrics (ROUGE, BLEU, METEOR, BERTScore, etc.) on each generated profile, run `python type1_hallucination.py`

This step produces measures in the following directories. For your convenience, we already ran the measurement creation step and provided the measurements in the `type_1_eval` directory.
```
type_1_eval/
    beauty/
        iterative/
            metrics.csv
        preference_negative/
            metrics.csv
        preference_positive/
            metrics.csv
        vanilla/
            metrics.csv
        vanilla_structured/
            metrics.csv
    movies/
        iterative/
            metrics.csv
        preference_negative/
            metrics.csv
        preference_positive/
            metrics.csv
        vanilla/
            metrics.csv
        vanilla_structured/
            metrics.csv
```
Here each `metrics.csv` file contains the following fields: 
```
rouge1_f,rouge2_f,rougeL_f,meteor,bleu1,bleu2,bleu3,bleu4,bertscore_precision,bertscore_recall,bertscore_f1,user_id,num_interactions
```

## Recommendation Generation

Run the following command to generate recommendations across all
datasets and profile types:

```bash
cd src
python automate_recommend.py
```

However, we recommend against running `automate_recommend.py` as there might be problems with OpenAI API rate limits if your account is not in a higher tier. An alternative is to
generate recommendations for a specific dataset, profile type, and subset
with the following command:

```bash
cd src
# example: python recommend.py All_Beauty iterative 1
python recommend.py dataset profile_type subset
```
Here `dataset` is either `All_Beauty` or `Movies_and_TV`, `profile_type` is one of `iterative`, `preference_negative`, `preference_positive`, `vanilla`, or `vanilla_structured`, and `subset` is from 1 to 10.

This command will produce the following files. 
For your convenience, we already ran the recommendation generation step 
and provided the recommendations in the `recommendations` directory.

```
recommendations/
    beauty/
        iterative/
            user_{id}.json
            ...
        preference_negative/
            ...
        preference_positive/
            ...
        vanilla/
            ...
        vanilla_structured/ 
            ...
    movies/
        iterative/
            ...
        preference_negative/
            ...
        preference_positive/
            ...
        vanilla/
            ...
        vanilla_structured/ 
            ...
```
Here each `user_{id}.json` file contains the following fields:
```
{
    "candidate": ...
    "binary_pos": ...
    "binary_neg": ...
}
```

## Type 2 Hallucination Measurement

Run the following command to measure type 2 hallucination across all
datasets and profile types:

```bash
cd src
python automate_evaluate_2.py
```

However, we recommend against running `automate_evaluate_2.py` as there might be problems with OpenAI API rate limits if your account is not in a higher tier or if you have limited cuda memory. An alternative is to generate recommendations for a specific dataset, profile type, and subset with the following command:

```bash
cd src
# example: python evaluate_2.py All_Beauty iterative 1
python evaluate_2.py dataset profile_type subset
```

This step produces measures in the following directories. For your convenience, we already ran the measurement step and provided the measurements in the `type_2_eval` directory.
```
type_2_eval/
    beauty/
        iterative/
            metrics.csv
        preference_negative/
            metrics.csv
        preference_positive/
            metrics.csv
        vanilla/
            metrics.csv
        vanilla_structured/
            metrics.csv
    movies/
        iterative/
            metrics.csv
        preference_negative/
            metrics.csv
        preference_positive/
            metrics.csv
        vanilla/
            metrics.csv
        vanilla_structured/
            metrics.csv
```
Here each `metrics.csv` file contains the following fields: 
```
user_id, true_positive,true_negative,accuracy,bertscore_precision,bertscore_recall,bertscore_f1,hit_rate_at_1,hit_rate_at_5,hit_rate_at_10,ndcg_at_1,ndcg_at_5,ndcg_at_10,consistency,nepc
```


## Correlation Measurement

To produce correlation measures, run the following command:

```bash
cd src
python compute_correlation.py
```

For your convenience, we already computed everything and provided the results in the `correlation` directory.

```
correlation/
    beauty/
        iterative/
            type_1_recommendation.csv
            type_1_type_2.csv
            type_2_recommendation.csv
        preference_negative/
            ...
        preference_positive/
            ...
        vanilla/
            ...
        vanilla_structured/
            ...
    movies/
        iterative/
            ...
        preference_negative/
            ...
        preference_positive/
            ...
        vanilla/
            ...
        vanilla_structured/
            ...
```
Here 
- `type_1_recommendation.csv` contains the correlation between type 1 hallucination measures and recommendation quality measures.
- `type_1_type_2.csv` contains the correlation between type 1 hallucination measures and type 2 hallucination measures.
- `type_2_recommendation.csv` contains the correlation between type 2 hallucination measures and recommendation quality measures.
