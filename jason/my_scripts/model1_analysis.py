import numpy as np
import os
import json
import pickle
import pandas
import glob
import matplotlib.pyplot as plt
plt.ion()

eval_names = [
    'Llama3_Baseline',
    'dolma_0p20_ehr_0p80',
    'dolma_0p05_ehr_0p95',
    'dolma_0p05_pubmedpapers_0p05_ehr_0p90',
]
all_cats = [
    "mcpqa","medmcqa","medqa_4options","mmlu","mmlu_humanities","mmlu_formal_logic","mmlu_high_school_european_history",
    "mmlu_high_school_us_history","mmlu_high_school_world_history","mmlu_international_law","mmlu_jurisprudence",
    "mmlu_logical_fallacies","mmlu_moral_disputes","mmlu_moral_scenarios","mmlu_philosophy","mmlu_prehistory",
    "mmlu_professional_law","mmlu_world_religions","mmlu_other","mmlu_business_ethics","mmlu_clinical_knowledge",
    "mmlu_college_medicine","mmlu_global_facts","mmlu_human_aging","mmlu_management","mmlu_marketing","mmlu_medical_genetics",
    "mmlu_miscellaneous","mmlu_nutrition","mmlu_professional_accounting","mmlu_professional_medicine","mmlu_virology",
    "mmlu_social_sciences","mmlu_econometrics","mmlu_high_school_geography","mmlu_high_school_government_and_politics",
    "mmlu_high_school_macroeconomics","mmlu_high_school_microeconomics","mmlu_high_school_psychology","mmlu_human_sexuality",
    "mmlu_professional_psychology","mmlu_public_relations","mmlu_security_studies","mmlu_sociology","mmlu_us_foreign_policy",
    "mmlu_stem","mmlu_abstract_algebra","mmlu_anatomy","mmlu_astronomy","mmlu_college_biology","mmlu_college_chemistry",
    "mmlu_college_computer_science","mmlu_college_mathematics","mmlu_college_physics","mmlu_computer_security","mmlu_conceptual_physics",
    "mmlu_electrical_engineering","mmlu_elementary_mathematics","mmlu_high_school_biology","mmlu_high_school_chemistry",
    "mmlu_high_school_computer_science","mmlu_high_school_mathematics","mmlu_high_school_physics","mmlu_high_school_statistics",
    "mmlu_machine_learning","pubmedqa",
]

med_cats = [
    "mmlu_professional_medicine", "mmlu_college_medicine","mmlu_clinical_knowledge","mmlu_virology",
]

gen_cats = [
    "mmlu_formal_logic","mmlu_high_school_european_history",
    "mmlu_high_school_us_history","mmlu_high_school_world_history","mmlu_international_law","mmlu_jurisprudence",
    "mmlu_logical_fallacies","mmlu_moral_disputes","mmlu_moral_scenarios","mmlu_philosophy","mmlu_prehistory",
    "mmlu_professional_law","mmlu_world_religions","mmlu_business_ethics","mmlu_global_facts","mmlu_management","mmlu_marketing",
    "mmlu_professional_accounting","mmlu_social_sciences","mmlu_econometrics","mmlu_high_school_geography","mmlu_high_school_government_and_politics",
    "mmlu_high_school_macroeconomics","mmlu_high_school_microeconomics","mmlu_professional_psychology","mmlu_public_relations","mmlu_security_studies",
    "mmlu_sociology","mmlu_us_foreign_policy","mmlu_abstract_algebra","mmlu_astronomy","mmlu_college_computer_science","mmlu_college_mathematics",
    "mmlu_college_physics","mmlu_computer_security","mmlu_conceptual_physics","mmlu_electrical_engineering","mmlu_elementary_mathematics",
    "mmlu_high_school_computer_science","mmlu_high_school_mathematics","mmlu_high_school_physics","mmlu_high_school_statistics",
    "mmlu_machine_learning",
]

cats = [
    "medmcqa", "medqa_4options", "pubmedqa", "mcpqa", "mmlu_anatomy",  "mmlu_clinical_knowledge",
    "mmlu_college_biology", "mmlu_college_medicine", "mmlu_medical_genetics", "mmlu_professional_medicine", 
    "mmlu"
]

eval_cats = [
    ["medmcqa", "medqa_4options"],
    ["pubmedqa"],
    ["mcpqa"],
    ["mmlu"],
]

def gen_df(
    results_fn='/Users/jasonwolfe/Documents/Cerebras/Mayo/Model-1/results/finetune_and_eval/all_results.pkl',
):
    frames = []
    data_mix_names = []
    with open(results_fn,'rb') as fid: 
        res = pickle.load(fid)
    for data_mix, data in res.items():
        data_mix_names.append(data_mix)
        metrics = np.concatenate([np.expand_dims(np.array(data['metrics'][mnm]),1) for mnm in all_cats], axis=1)
        ind = pandas.MultiIndex.from_product([[data_mix], data['checkpoint_step']], names=['run_name','step'])
        _df = pandas.DataFrame(metrics, columns=all_cats, index=ind)
        frames.append(_df)
    df = pandas.concat(frames)
    return df
    
def gen_means(df):
    base = df.query('run_name == "Llama3_Baseline"')[gen_cats].mean(axis=1).to_frame("General")
    base["Medical"] = df.query('run_name == "Llama3_Baseline"')[med_cats].mean(axis=1).to_frame()
    base_mns = base.groupby("run_name").mean()
    base_sds = base.groupby("run_name").std()

    res = df.query('step >= 2034')[gen_cats].mean(axis=1).to_frame("General")
    res["Medical"] = df.query('step >= 2034')[med_cats].mean(axis=1).to_frame()
    res_mns = res.groupby("run_name").mean()
    res_sds = res.groupby("run_name").std()

    mns = pandas.concat([base_mns, res_mns])
    sds = pandas.concat([base_sds, res_sds])
    return mns, sds

def plot_eval_results(
        results_fn='/Users/jasonwolfe/Documents/Cerebras/Mayo/Model-1/results/finetune_and_eval/all_results.pkl',
        output_dir='/Users/jasonwolfe/Documents/Cerebras/Mayo/Model-1/results/finetune_and_eval/model_1_doc_figures',
):
    df = gen_df(results_fn)
    for cat in eval_cats:
        df[cat].mean(axis=1).unstack(level=0).plot()
        # for nm in data_mix_names:
        #     df[cat][nm].plot()
        # plt.legend(data_mix_names)
        plt.title(','.join(cat))
    mns = df.query('step >= 2034')[['medmcqa','medqa_4options','mcpqa','pubmedqa','mmlu']].groupby('run_name').mean()
    sd = df.query('step >= 2034')[['medmcqa','medqa_4options','mcpqa','pubmedqa','mmlu']].groupby('run_name').std()
    mns.T.plot.bar(yerr=sd.T,rot=0, figsize=[12,7]) 
    plt.ylim([0.5,0.77])
    fn = os.path.join(output_dir,"mmqa_acc_vs_datamix.png")
    plt.savefig(fn)
    # for nm in eval_names:
    #     plt.figure()
    #     df.loc([nm])[['medmcqa', 'medqa_4options']].mean(axis=1).plot(marker='.')
    #     df['mmlu'].plot(marker='.')
    #     plt.legend('MultiMedQA + MCPQA', 'MMLU')  
    #     plt.title(nm)
    return df

def plot_mmlu(
        df=None,
        output_dir='/Users/jasonwolfe/Documents/Cerebras/Mayo/Model-1/results/finetune_and_eval/model_1_doc_figures',
    ):
    if df is None:
        df = gen_df()
    mns, sds = gen_means(df)
    # base = df.query('run_name == "Llama3_Baseline"')[gen_cats].mean(axis=1).to_frame("General")
    # base["Medical"] = df.query('run_name == "Llama3_Baseline"')[med_cats].mean(axis=1).to_frame()
    # base_mns = base.groupby("run_name").mean()
    # base_sds = base.groupby("run_name").std()

    # res = df.query('step >= 2034')[gen_cats].mean(axis=1).to_frame("General")
    # res["Medical"] = df.query('step >= 2034')[med_cats].mean(axis=1).to_frame()
    # res_mns = res.groupby("run_name").mean()
    # res_sds = res.groupby("run_name").std()

    # mns = pandas.concat([base_mns, res_mns])
    # sds = pandas.concat([base_sds, res_sds])
    mns.plot.bar(yerr=sds,rot=0)
    plt.ylim([0.61,0.68])
    plt.gca().set_xticklabels(['Llama3-8B Baseline','Dolma: 5%, EHR: 95%, PubMed: 0%','Dolma: 5%, EHR: 90%, PubMed: 5%','Dolma: 20%, EHR: 80%, PubMed: 0%'])
    plt.ylabel("Accuracy")   
    plt.xlabel('Domain Adaption Data Mixes') 
    plt.gcf().set_figwidth(15) 
    fn = os.path.join(output_dir,'acc_vs_data_mix_plot.png')
    plt.savefig(fn)

    # mns = df.query("step>=2034")[gen_cats].mean(axis=0).to_frame("General")
    # sds = df.query("step>=2034")[gen_cats].std(axis=0).to_frame("General")

    # mns["Medical"] = df.query("step>=2034")[med_cats].mean(axis=0).mean()
    # sds["Medical"] = df.query("step>=2034")[med_cats].std(axis=0).to_frame()

    f0 = df.query('run_name=="Llama3_Baseline"')[gen_cats].mean(axis=0).to_frame().mean().to_frame("General Domain")
    f0["Medical Domain"] = df.query('run_name=="Llama3_Baseline"')[med_cats].mean(axis=0).to_frame().mean().to_frame()
    f0 = f0.rename(index={0:"Baseline"})
    f1 = df.query("step>=2034")[gen_cats].mean(axis=0).to_frame().mean().to_frame("General Domain")
    f1["Medical Domain"] = df.query("step>=2034")[med_cats].mean(axis=0).to_frame().mean().to_frame()
    f1 = f1.rename(index={0:"Domain Adapted"})

    s0 = get_std(df,gen_cats,'run_name=="Llama3_Baseline"',"General Domain")
    s0["Medical Domain"] = get_std(df,med_cats,'run_name=="Llama3_Baseline"',"Medical Domain")
    s0 = s0.rename(index={0:"Baseline"})

    s1 = get_std(df,gen_cats,"step>=2034","General Domain")
    s1["Medical Domain"] = get_std(df,med_cats,"step>=2034","Medical Domain")
    s1 = s1.rename(index={0:"Domain Adapted"})

    fs = pandas.concat([f0,f1], axis=0)
    ss = pandas.concat([s0,s1], axis=0)
    fs.T.plot.bar(yerr=ss.T, rot=0)
    plt.ylim([.62,.67])
    plt.ylabel("Accuracy")   
    fn = os.path.join(output_dir,'medical_vs_general_bar.png')
    plt.savefig(fn)
    # return mns, base, res, med_cats, gen_cats
    return mns, med_cats, gen_cats

def get_std(df, cats, query, column_name):
    new_df = df.query(query)[cats].std(axis=0)**2
    new_df = pandas.DataFrame({column_name: [new_df.sum()]})
    new_df = (new_df**0.5)/len(cats)
    return new_df

def plot_acc_vs_ratio(
        df=None,
        output_dir='/Users/jasonwolfe/Documents/Cerebras/Mayo/Model-1/results/finetune_and_eval/model_1_doc_figures',
    ):
    if df is None:
        df = gen_df()
    mns, sds = gen_means(df)
    mns['EHR Percent']=[0,95,90,80]
    mns.sort_values('EHR Percent').plot(x="EHR Percent",y="General",marker='o', linestyle='none',title='General MMLU vs. EHR Percent')
    fn = os.path.join(output_dir,'general_vs_ehr_percent_plot.png')
    plt.savefig(fn)
    mns.sort_values('EHR Percent').plot(x="EHR Percent",y="Medical",marker='o', linestyle='none',title='Medical MMLU vs. EHR Percent')
    fn = os.path.join(output_dir,'medical_vs_ehr_percent_plot.png')
    plt.savefig(fn)


def plot_bar(df=None):
    if df == None:
        df = plot_eval_results()
    # for cat in cats:
    for cat in ["medmcqa", "medqa_4options", "pubmedqa", "mcpqa", "mmlu"]:
        dicts = []
        steps = []
        all_vals = []
        for nm in eval_names:
            try:
                d = df[cat][nm].to_dict()
            except:
                import pdb; pdb.set_trace()
            dicts.append(d)
            steps.extend(list(d.keys()))
            vals = list(d.items())
            all_vals.append(vals)
        steps = np.unique(steps)
        for step in steps:
            for d in dicts:
                if step not in d.keys():
                    d[step] = -1
        new_dict = {'acc': [], 'std': []}
        steps = [1695,2034, 2373, 2712]
        for d, name in zip(dicts, eval_names):
            m = np.mean([d[step] for step in steps])
            v = np.std([d[step] for step in steps])
            new_dict['acc'].append(m)
            new_dict['std'].append(v)
        df_out = pandas.DataFrame(new_dict, index=eval_names)
        df_out.plot.bar(y='acc', yerr='std', rot=0)
        plt.ylim([0.5,0.75])  
        plt.title(cat)
        plt.gca().set_xticklabels(["20% Dolma\n80% EHR","5% Dolma\n95% EHR","5% Dolma\n5% Pubmed\n90% EHR"])   
    return dicts, df_out

def plot_loss_curves(data_dir):
    fns = glob.glob(os.path.join(data_dir,"*.json"))
    plt.figure()
    labels = []
    for fn in fns:
        labels.append(os.path.basename(fn).split('_..json')[0])
        loss = []
        step = []
        with open(fn,'r') as fid:
            y = json.load(fid)
        for yi in y:
            step.append(yi[1])
            loss.append(yi[2])
        plt.plot(step,loss)
    plt.legend(labels)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.savefig(os.path.join(data_dir,"loss_curves.png"))
