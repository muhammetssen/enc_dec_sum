import pandas as pd
from pathlib import Path
from numpy import random
random.seed(35)

idx=random.randint(10000, size=(50))
print(idx)
home_path = Path("/Users/bbaykara/Desktop/results")

'''
dataset = "tr_news"
task = "summary"
models = ["mt5", "mbart", "berturk32k", "berturk32k_cased", "mbert_uncased", "mbert_cased"]
checkpoints = ["checkpoint-60718","checkpoint-43370","checkpoint-52050","checkpoint-52050","checkpoint-69400","checkpoint-78075"]

dataset = "tr_news"
task = "title"
models = ["mt5", "mbart", "berturk32k", "berturk32k_cased", "mbert_uncased", "mbert_cased"]
checkpoints = ["checkpoint-52044","checkpoint-34696","checkpoint-52044","checkpoint-34696","checkpoint-52044","checkpoint-60718"]

dataset = "ml_sum"
task = "title"
models = ["mt5", "mbart", "berturk32k", "berturk32k_cased", "mbert_uncased", "mbert_cased"]
checkpoints = ["checkpoint-38950","checkpoint-31160","checkpoint-38950","checkpoint-31160","checkpoint-38950","checkpoint-38950"]

'''

dataset = "ml_sum"
task = "summary"
models = ["mt5", "mbart", "berturk32k", "berturk32k_cased", "mbert_uncased", "mbert_cased"]
checkpoints = ["checkpoint-62320","checkpoint-15580","checkpoint-46740","checkpoint-31160","checkpoint-46740","checkpoint-62320"]



results = {}
for model, checkpoint in zip(models,checkpoints):
    path = home_path.joinpath(dataset, task , dataset + "_" + model + "_" + task ,checkpoint,"text_outputs.csv")
    results[model] = pd.read_csv(path).iloc[idx]


for i in idx:
    print("**************************************************************")
    print("ID",i)
    print("------------------------ SOURCE ------------------------")
    print(results["mt5"].loc[i]['source'])
    print("------------------------ REFERENCE ------------------------")
    print(results["mt5"].loc[i]['target'])
    for model in models:
        print("------------------------"+model+"------------------------")
        print(results[model].loc[i]['predictions'])
        print()

