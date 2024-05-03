import argparse
import logging
import os
import shutil
import pickle
import glob
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import tqdm


def remove_lines(fn, output_dir, lines_to_remove):
    idx = 0
    fn_out = os.path.join(output_dir, os.path.basename(fn))
    with open(fn,'r') as f_in, open(fn_out,'w') as f_out:
        for line in f_in:
            if idx not in lines_to_remove:
                f_out.write(line)
            idx += 1

def deduplicate(args):
    logging.info("Processing duplicates...")

    # load pickled components and other artifacts
    with open(args.input_file, "rb") as fin:
        components, n_components, reversed_mapper = pickle.load(fin)

    duplicates = defaultdict(set)
    n_duplicate_docs = 0
    
    for component in tqdm.tqdm(components):
        for j in range(1, len(component)):
            doc = reversed_mapper[component[j]]
            file_name, doc_idx = doc.split("@")
            file_name = os.path.join(args.input_dir, file_name.split('/')[-1])
            duplicates[file_name].add(int(doc_idx))
            n_duplicate_docs += 1

    logging.info(
        f"Number of documents with duplicate lines is: {n_duplicate_docs}"
    )

    logging.info("Generating final dataset...")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    # Copy non-duplicate files to output_dir
    files = glob.glob(os.path.join(args.input_dir,f'*.{args.format}'))
    for file in files:
        if file not in duplicates.keys():
            new_file = os.path.join(args.output_dir,os.path.basename(file))
            shutil.copy(file, new_file)

    # Process files with duplicate entries
    with Pool(processes=cpu_count()) as pool:
        pool.starmap(
            remove_lines,
            [
                (file, args.output_dir, line_numbers)
                for (file, line_numbers) in duplicates.items()
            ],
        )




def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_file",
        type=str,
        help="The connected_components.pickle file from previous stage",
        required=True,
    )
    parser.add_argument(
        "--input_dir", type=str, help="Input directory", required=True
    )
    parser.add_argument(
        "--output_dir", type=str, help="Output directory", required=True
    )
    parser.add_argument("--format", type=str, help="File format of the dataset")
    args = parser.parse_args()
    generate_duplicates(args)


if __name__ == "__main__":
    main()
