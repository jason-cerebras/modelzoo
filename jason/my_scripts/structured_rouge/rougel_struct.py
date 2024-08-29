import numpy as np
import random
import copy
from rouge_score import rouge_scorer
import matplotlib.pyplot as plt
plt.ion()
from tqdm import tqdm

def gen_test_set(
        vocab_file='/Users/jasonwolfe/Documents/Cerebras/git/modelzoo_mayo_fork/modelzoo/modelzoo/transformers/utils/structured_rouge/llama2_vocab.txt',
        max_num=6,
        num_list_pairs=2,
        num_words_per_sent=10,
        perc_noise=0.3,
        max_diff=3,
):
    """
    - Generates a test dataset that can be used with the `calc_rougel_struct` function.
    - one `sample` is a pair of lists, with each list containing up to `max_num` synthetic sentences. 
      The sentences are created by randomly sampling from the `vocab_file`.
        - the number of sentences in each list can differ by up to `max_diff`
        - the sentences in each list of the pair are created to be identical. Each sentence can be perturbed by substituting
          in random words from the `vocab_file`. The number of words that get substituted can be controled with `perc_noise`
        - the goal of the above 2 perturbations is to mimic the case when one impressions list has more bullet points than 
          the other and each sentence in the list can be different
    - Creates a dictionary of two lists. Each element in the list is a `sample`
    Inputs:
    vocab_file: path to text file containing set of words to be used to generate the synthetic sentences
    max_num: Maximum number of sentences to inclide in each list
    num_list_pairs: number of pairs of lists to create
    num_words_per_sent: length of the synthetic sentences
    perc_noise: percentage of words to replace in the sentences to make the sentences in the lists different
    max_diff: maximum difference in the number of sentences contained in each list
    """
    with open(vocab_file,'r') as fid:
        vocab = [s.strip() for s in fid]

    pairs = {'list1':[], 'list2':[]}
    for ii in range(num_list_pairs):
        full_list = []
        num_sentences_list1 = random.sample(range(1, max_num), 1)[0]
        if max_diff > 0:
            d_num = random.sample(range(-max_diff, max_diff), 1)[0]
        else:
            d_num = 0
        num_sentences_list2 = max([min([num_sentences_list1+d_num,max_num]),1])
        for jj in range(max_num):
            word_indices = random.sample(range(0, len(vocab)), num_words_per_sent)
            sent = ' '.join([vocab[idx] for idx in word_indices])
            full_list.append(sent)
        list1 = full_list[0:num_sentences_list1]
        list2 = full_list[0:num_sentences_list2]
        if perc_noise > 0:
            list2_w_noise = []
            for s in list2:
                sp, n = add_noise(s, vocab, perc_noise)
                list2_w_noise.append(sp)
            list2 = list2_w_noise
        random.shuffle(list1)
        random.shuffle(list2)
        pairs['list1'].append(list1)
        pairs['list2'].append(list2)
    return pairs, vocab

def add_noise(sent, vocab, perc=0.2):
    words = sent.split(' ')
    n_tot = len(words)
    n_replace = int(perc*n_tot)
    inds = random.sample(list(np.arange(n_tot)),n_replace)
    for ind in inds:
        words[ind] = random.sample(vocab, 1)[0]
    new_sent = ' '.join(words)
    return new_sent, n_replace


def calc_rougel_struct(list1, list2):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)  
    if len(list1) > len(list2):
        long_list = list1
        short_list = list2
    else:
        long_list = list2
        short_list = list1     
    res = np.zeros([len(long_list),len(short_list)])    
    for ii in range(len(long_list)):
        for jj in range(len(short_list)):
            res[ii,jj] = scorer.score(long_list[ii],short_list[jj])['rougeL'].fmeasure
    sc = np.mean([max(res[ii,:]) for ii in range(res.shape[0])])
    return res, sc

def calc_rougel_unstruct(list1, list2):
    s1 = '. '.join(list1)
    s2 = '. '.join(list2)
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True) 
    sc = scorer.score(s1,s2)['rougeL'].fmeasure
    return sc


def calc_rougel_struct_all(pairs):
    scores = []
    for list1, list2 in zip(pairs['list1'], pairs['list2']):
        r, sc = calc_rougel_struct(list1, list2)
        scores.append(sc)
    return np.mean(scores)

def calc_rougel_unstruct_all(pairs):
    scores = []
    for list1, list2 in zip(pairs['list1'], pairs['list2']):
        sc = calc_rougel_unstruct(list1, list2)
        scores.append(sc)
    return np.mean(scores)

def run_perc_diff_test(
    max_num=8,
    num_list_pairs=100,
    num_words_per_sent=12,
    perc_noise_values=np.arange(0,1.02,.05),
    max_diff_value=3,
    vocab_file='/Users/jasonwolfe/Documents/Cerebras/git/modelzoo_mayo_fork/modelzoo/modelzoo/transformers/utils/structured_rouge/llama2_vocab.txt'
):
    """
    This can be used to generate synthetic datasets with increasing differences between the sentences. Creates a plot of `perc_noise`
    versus `rougel_struct` score. For reference, also creates a plot of the unstructured rouge-l score which, instead of comparing 
    sentence-by-sentence, combines all the sentences in a list into one text and calculates the rouge-l score for the combined text.
    """
    struct_rougel = []
    unstruct_rougel = []
    for p in tqdm(perc_noise_values):
        pairs,v = gen_test_set(
            vocab_file=vocab_file,
            max_num=max_num,
            num_list_pairs=num_list_pairs,
            num_words_per_sent=num_words_per_sent,
            perc_noise=p,
            max_diff=max_diff_value,
        )
        struct_rougel.append(calc_rougel_struct_all(pairs))
        unstruct_rougel.append(calc_rougel_unstruct_all(pairs))
    plt.figure()
    plt.plot(perc_noise_values,struct_rougel,'b.')
    plt.plot(perc_noise_values,unstruct_rougel,'r.')
    plt.legend(["structured","unstructured"])
    plt.title('Structured vs. Unstructured RougeL versus Percent List Difference')


def run_num_diff_test(
    max_num=8,
    num_list_pairs=200,
    num_words_per_sent=14,
    perc_noise_value=0.1,
    max_diff_values=np.arange(0,12),
    vocab_file='/Users/jasonwolfe/Documents/Cerebras/git/modelzoo_mayo_fork/modelzoo/modelzoo/transformers/utils/structured_rouge/llama2_vocab.txt'
):
    """
    This can be used to generate synthetic datasets with increasing differences in the number of sentences in each list. Creates a plot 
    of `max_diff_values` versus `rougel_struct` score. For reference, also creates a plot of the unstructured rouge-l score which, instead of comparing 
    sentence-by-sentence, combines all the sentences in a list into one text and calculates the rouge-l score for the combined text.
    """
    struct_rougel = []
    unstruct_rougel = []
    for p in tqdm(max_diff_values):
        pairs,v = gen_test_set(
            vocab_file=vocab_file,
            max_num=max_num,
            num_list_pairs=num_list_pairs,
            num_words_per_sent=num_words_per_sent,
            perc_noise=perc_noise_value,
            max_diff=p,
        )
        struct_rougel.append(calc_rougel_struct_all(pairs))
        unstruct_rougel.append(calc_rougel_unstruct_all(pairs))
    plt.figure()
    plt.plot(max_diff_values,struct_rougel,'b.')
    plt.plot(max_diff_values,unstruct_rougel,'r.')
    plt.legend(["structured","unstructured"])
    plt.title('Structured vs. Unstructured RougeL versus Number List Difference')





    