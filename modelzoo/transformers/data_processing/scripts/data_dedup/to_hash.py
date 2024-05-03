import argparse
import gc
import logging
import os
import pickle
import re
import string
import sys
import threading
import time
from collections import deque
from itertools import repeat
from multiprocessing import Lock, Pool, cpu_count

import ftfy
from datasketch import MinHash
from more_itertools import chunked
from nltk import ngrams

# MZ: sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
# from cerebras.modelzoo.data_preparation.nlp.hdf5_preprocessing.utils import (
#     Reader,
# )
from modelzoo.transformers.data_processing.scripts.hdf5_preprocessing.utils import (
    Reader
)


dq = deque()

def print_docs_processed():
    while True:
        if dq:
            logging.info(f"\nDocuments processed: {dq[-1]}")
        time.sleep(60)


def custom_progress_bar(animation_delay=0.5):
    chars = ['|', '/', '-', '\\']
    progress = 0

    doc_printer_thread = threading.Thread(target=print_docs_processed)
    doc_printer_thread.daemon = True
    doc_printer_thread.start()

    while True:
        sys.stdout.write(f'\rProcessing: [{chars[progress % len(chars)]}]')
        sys.stdout.flush()
        progress += 1
        time.sleep(animation_delay)


def preprocess_string(s):
    # Lowercase the input string
    s = s.lower()
    # Remove punctuation
    s = s.translate(str.maketrans("", "", string.punctuation))
    # Remove consecutive spaces, newlines, tabs in the middle and in the beginning / end
    s = re.sub(r"\s+", " ", s.strip())
    return s


def get_features(s, width):
    s = preprocess_string(s)
    return map(lambda x: "".join(x), ngrams(s, width))


def clean(s):
    return preprocess_string(s)


def get_documents(input_dir, jsonl_key, format, threshold, job_id, n_jobs):
    files = []
    gc.collect()
    all_files = []

    for root, dirs, files in os.walk(input_dir):
        for file in files:
            all_files.append(os.path.join(root, file))

    for file in all_files:
        parts = file.split('.')[1:]
        file_format = '.'.join(parts)
        if format == file_format:
            files.append(os.path.basename(file))

    no_of_files = len(files)
    start = job_id*n_jobs
    end = start + (no_of_files // n_jobs)

    for index in range(start,end):
        input_file = files[index]
        file_path = os.path.join(input_dir, input_file)
        tokenizable_columns = {"jsonl_key": jsonl_key}
        reader = Reader(file_path, tokenizable_columns)

        for doc_id, doc in enumerate(reader.stream_data()):
            if len(clean(doc)) > threshold:
                yield doc, file_path, doc_id


def to_minhash(chunks):
    gc.collect()
    buckets = []
    documents, output_dir, width, dataset_name = chunks
    for doc in documents:
        text, file_path, doc_id = doc[0], doc[1], doc[2]
        file_name = file_path.split("/")[-1]
        output_name = f"{dataset_name}/{file_name}"

        text = ftfy.fix_text(text, normalization="NFC")

        m = MinHash(num_perm=128)
        m.update_batch(map(lambda x: x.encode('utf8'), get_features(text, width)))
        buckets.append(
            {"file_name": output_name, "doc_id": doc_id, "hash": m,}
        )
    return buckets


def output_results(output_dir, results, chunk_id, iter):
    with open(
        f"{output_dir}/minhash_nfc/{iter}-{chunk_id}.pickle", "wb"
    ) as fout:
        pickle.dump(results, fout)


def generate_hashes(args):
    lock = Lock()
    docs_processed = 0
    if not os.path.exists(f"{args.output_dir}/minhash_nfc"):
        os.mkdir(f"{args.output_dir}/minhash_nfc")

    documents = get_documents(
        args.input_dir, args.jsonl_key, args.format, args.threshold, args.job_id, args.n_jobs
    )

    results = []
    chunk_id = 0
    gc.collect()
    with Pool(processes=cpu_count()) as pool:
        results_iterator = pool.imap(
            to_minhash,
            zip(
                chunked(documents, args.batch_size),
                repeat(args.output_dir),
                repeat(args.window_size),
                repeat(args.dataset_name),
            ),
        )

        for i, chunks in enumerate(results_iterator):
            for chunk in chunks:
                if len(results) == args.batch_size:
                    with lock:
                        docs_processed += args.batch_size
                        dq.append(docs_processed)
                    output_results(
                        args.output_dir, results, chunk_id, args.job_id
                    )
                    del results
                    gc.collect()
                    results = []
                    chunk_id += 1
                results.append(chunk)

    if results:
        with lock:
            docs_processed += len(results)
        logging.info(f"\nFinal document count: {docs_processed}")
        output_results(args.output_dir, results, chunk_id, args.job_id)


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

    args = parser.parse_args()

    progress_thread = threading.Thread(target=custom_progress_bar)
    progress_thread.daemon = True
    progress_thread.start()

    generate_hashes(args)
