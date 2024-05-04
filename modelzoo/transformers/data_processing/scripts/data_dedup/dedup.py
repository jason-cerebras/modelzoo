import os
import sys
import time
import json
import shutil
import threading
import logging
import argparse
import random
import numpy as np
import deduplicate_jsonl
import generate_connected_components
import generate_duplicate_pairs
import to_hash



def custom_progress_bar(length=30, animation_delay=0.5):
    chars = ['|', '/', '-', '\\']
    progress = 0

    while True:
        sys.stdout.write(f'\rProcessing: [{chars[progress % len(chars)]}]')
        sys.stdout.flush()
        progress += 1
        time.sleep(animation_delay)

def dedup_dataset(args):
    #python to_hash.py --dataset_name <dataset-name> --input_dir <input-dir> --output_dir <output-dir> --job_id <job-id> --jsonl_key <jsonl-key> --format <format-of-the-dataset>
    progress_thread = threading.Thread(target=custom_progress_bar)
    progress_thread.daemon = True
    progress_thread.start()
    to_hash.generate_hashes(args)
    
    #python generate_duplicate_pairs.py --input_dir <output-directory-from-previous-step> --out_file <output-directory>/duplicates/duplicate_pairs.txt
    input_dir = args.input_dir
    args.input_dir = args.output_dir
    if not os.path.exists(os.path.join(args.output_dir, 'duplicates')):
        os.mkdir(os.path.join(args.output_dir, 'duplicates'))
    args.out_file = os.path.join(args.output_dir, 'duplicates', 'duplicate_pairs.txt')
    progress_thread = threading.Thread(target=custom_progress_bar)
    progress_thread.daemon = True
    progress_thread.start()
    generate_duplicate_pairs.generate_pairs(args)

    #python generate_connected_components.py --input_dir <output-directory-from-previous-step}/duplicates --out_file <output-directory>/duplicates/connected_components.pickle
    args.input_dir = os.path.join(args.output_dir, 'duplicates')
    args.out_file = os.path.join(args.output_dir,'connected_components.pickle')
    generate_connected_components.generate_connected_components_mp(args)

    #python deduplicate_dataset.py --input_file <output-directory-from-previous-step>/duplicates/connected_components.pickle --input_dir <input-dir> --output_dir <final-output-dir> --format <format-of-dataset>
    args.input_file = args.out_file
    args.input_dir = input_dir
    args.output_dir = os.path.join(args.output_dir,"deduplicated_output")
    deduplicate_jsonl.deduplicate(args)


def gen_test_dataset(
    output_dir,
    num_docs=10,
    num_paragraphs_per_doc=10,
    num_words_per_paragraph=100,
    min_num_copied_paragraphs_per_doc=0,
    max_num_copied_paragraphs_per_doc=3,
    min_copied_paragraph_overlap=1.0, # up to 1
    vocab_file='/cb/home/jason/ws/git/monolith/src/models/src/cerebras/modelzoo/data_preparation/nlp/data_dedup/test2/vocab.txt',
):
    assert num_paragraphs_per_doc > max_num_copied_paragraphs_per_doc, "max_num_copied_paragraphs_per_doc must be \
        less than num_paragraphs_per_doc"

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    min_copied_paragraph_nonoverlap = (1 - min_copied_paragraph_overlap)/2 #for every one that's copied two are the same. 

    with open(vocab_file,'r') as fid:
        vocab = [s.strip() for s in fid]
    
    for doc_num in range(num_docs):
        fn_out = os.path.join(output_dir,f'test_file_{doc_num}.jsonl')
        paragraphs = []
        for paragraph_num in range(num_paragraphs_per_doc):
            word_indices = random.sample(range(0, len(vocab)), num_words_per_paragraph)
            paragraph = ' '.join([vocab[idx] for idx in word_indices])
            paragraphs.append(paragraph)
        
        paragraphs_to_copy = random.randint(
            min_num_copied_paragraphs_per_doc,
            max_num_copied_paragraphs_per_doc
        )
        paragraph_inds = list(range(0, num_paragraphs_per_doc))
        paragraph_inds_to_replace = random.sample(paragraph_inds, paragraphs_to_copy)


        _ = [paragraph_inds.pop(paragraph_inds.index(ind)) for ind in paragraph_inds_to_replace]
        paragraph_inds_to_copy = random.sample(paragraph_inds, paragraphs_to_copy)

        for from_ind, to_ind in zip(paragraph_inds_to_replace, paragraph_inds_to_copy):
            num_words_to_replace = int((min_copied_paragraph_nonoverlap)*num_words_per_paragraph)

            # These are the word indices that will get replaced so that after copying these won't
            # match the original copied paragraph
            word_to_replace_inds = random.sample(range(0,num_words_per_paragraph), num_words_to_replace)

            paragraph_to_copy = paragraphs[from_ind].split(' ') #this is the paragraph that will get copied

            for ind in word_to_replace_inds:
                word_to_replace = paragraph_to_copy[ind]
                vocab_inds = list(np.arange(0,len(vocab)))
                _ = vocab_inds.pop(vocab.index(word_to_replace))
                new_ind = random.sample(vocab_inds,1)[0]
                paragraph_to_copy[ind] = vocab[new_ind]
            paragraphs[to_ind] = ' '.join(paragraph_to_copy)
        
        with open(fn_out,'w') as fid:
            for paragraph in paragraphs:
                j = json.dumps({"text": paragraph.encode('ascii', 'ignore').decode("utf-8")})
                fid.write(j + '\n')


    return paragraphs


# if __name__ == '__main__':
#     gen_test_dataset('./test_dataset')



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, help="Name of the dataset being processed.", required=True)
    parser.add_argument("--input_dir", type=str, help="Input directory which contains documents.", required=True)
    parser.add_argument(
        "--output_dir", type=str, help="Output directory to output MinHash files to.", required=True
    )
    parser.add_argument("--job_id", type=int, help="Job ID",default=0, required=False)
    parser.add_argument("--jsonl_key", type=str, default="text", help="JSONL key for the dataset", required=False)
    parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        help="Format of the dataset that needs to be processed.",
        required=False
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=0,
        help="Minimum size of documents that need to be present.",
        required=False
    )
    parser.add_argument(
        "--window_size", type=int, default=6, help="Window size", required=False
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100,
        help="Number of batches to output with.",
        required=False,
    )
    parser.add_argument(
        "--docs_per_core",
        type=int,
        default=1000,
        help="Number of documents that will be processed by each core.",
        required=False
    )
    parser.add_argument(
        "--n_jobs",
        type=int,
        default=1,
        help="Number of jobs to be spawned for parallel execution",
        required=False
    )
    parser.add_argument(
        "--jaccard_threshold",type=float,help="Threshold for Jaccard similarity", default=0.8, required=False
    )
    parser.add_argument(
        "--processes", type=int,help="Number of processes to parallelise on", default=1, required=False 
    ) 
    parser.add_argument(
        "--keep_first", action="store_false",help="when this flag is set the last duplicated instance is kept. Otherwise the first \
            instance is kept" 
    ) 
    args = parser.parse_args()
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    
    dedup_dataset(args)
