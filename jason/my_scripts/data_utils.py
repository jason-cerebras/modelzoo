import numpy as np
import tqdm
import h5py
import glob
import os
import matplotlib.pyplot as plt
import random
from transformers import AutoTokenizer

def count_tokens_h5(h5_dir, text_fieldname="data"):
    fns = glob.glob(os.path.join(h5_dir,"*.h5"))
    tot_tokens = 0
    for ii in tqdm.tqdm(range(len(fns))):
        fn = fns[ii]
        fid = h5py.File(fn,'r')
        y = fid[text_fieldname]
        num_tokens = y.shape[0]*y.shape[2]
        tot_tokens += num_tokens
    print(tot_tokens)
    return tot_tokens

def get_ratios(data_dir, text_fieldname="data"):
    token_counts = {}
    tot = 0
    for d in os.listdir(data_dir):
        _d = os.path.join(data_dir,d)
        t = count_tokens_h5(_d, text_fieldname=text_fieldname)
        token_counts[d] = t
        tot += t
    ratios = {k:v/tot for k,v in token_counts.items()}
    return token_counts, ratios

def plot_eval(
    dir="/cra-165/runs/2024-03-07_7B_pretrain/logs/", 
    output_dir="/cra-165/runs/2024-03-07_7B_pretrain/model_dir/finetune_1/eval_runs/results",
):
    checkpoint_steps = [0,2010,4020,6030,8040,10050,12060,14070,16750,19103]
    datasets = ["papers","abstracts","guidelines","mimic","slimpajama","mayo"]
    metrics = ["perplexity","accuracy","loss"]

    res = {k:{m:[] for m in metrics} for k in datasets}

    for dataset in datasets:
        for step in checkpoint_steps:
            fn = f"finetune_1_eval_checkpoint_{step}_{dataset}.out"
            with open(os.path.join(dir,fn), 'r') as fid:
                for line in fid:
                    k = "Metric: eval/lm_perplexity = "
                    if k in line:
                        m = float(line.split(k)[-1])
                        res[dataset]["perplexity"].append(m)
                    k = "Metric: eval/accuracy = "
                    if k in line:
                        m = float(line.split(k)[-1])
                        res[dataset]["accuracy"].append(m)
                    k = "Avg Eval Loss: "
                    if k in line:
                        m = float(line.split(k)[-1])
                        res[dataset]["loss"].append(m)
    clrs = "bgrkm"
    for metric in metrics:
        plt.figure()
        fig_fn = os.path.join(output_dir,f"{metric}_eval.png")
        for dataset, clr in zip(datasets, clrs):
            y = res[dataset][metric]
            plt.plot(checkpoint_steps,y,f"{clr}.-")
        plt.savefig(fig_fn)
    return res

def visualize_sample(data_dir, tokenizer_dir=None):
    if tokenizer_dir == None:
        t = AutoTokenizer.from_pretrained(
            "meta-llama/Llama-2-7b-chat-hf",
            token="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",
        )
    else:
        t = AutoTokenizer.from_pretrained(tokenizer_dir)
    all_files = glob.glob(os.path.join(data_dir, "*.h5"))
    current_file = random.choice(all_files)
    with h5py.File(current_file, "r") as f:
        data = f["data"]
        i = random.randrange(len(data) - 1)
        inp_ids = data[i,0]
        tgt_ids = data[i, 2]
        inp = t.decode(inp_ids)
        mask = data[i,1]
        target = t.decode(tgt_ids)

    sample_source = f"sample {i} from file {current_file}"
    return sample_source, [inp, mask, target, inp_ids, tgt_ids], t

