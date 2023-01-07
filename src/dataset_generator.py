import argparse
import pickle
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered_items', type=str, required=True, help='Path to filtered items file [csv format]]')
    parser.add_argument('--link_annotations', type=str, required=True, help='Path to link annotated text file [json format]]')
    parser.add_argument('--page_map', type=str, required=True, help='Path to page to item map file [csv format]')
    parser.add_argument('--output', type=str, required=True, help='Path where to save dataset [pickle format]]')
    args = parser.parse_args()

    FILTERED_ITEMS_PATH = Path(args.filtered_items)
    LINK_ANNOTATIONS_PATH = Path(args.link_annotations)
    PAGE_MAP_PATH = Path(args.page_map)
    DATASET_FOLDER_PATH = Path(args.output)

    CHUNKSIZE = 1000

    # Load filtered items
    print("Loading files...")
    df_filtered = pd.read_csv(FILTERED_ITEMS_PATH).copy()
    df_link_annotations = pd.read_json(LINK_ANNOTATIONS_PATH, lines=True, chunksize=CHUNKSIZE)
    df_page_map = pd.read_csv(PAGE_MAP_PATH, index_col=0)

    # Get size of df_link_annotations
    df_link_annotations_no_lines = sum(1 for line in open(LINK_ANNOTATIONS_PATH))

    # Get all qids from filtered items
    all_qids = set(df_filtered['qid'].values)
    all_qids.update(df_filtered["Work_of_art"].values)

    del df_filtered

    # Get all pages from filtered items
    all_pages = set(df_page_map.index.values)

    # filtered and mapped data
    dataset = []

    # Filter out pages that are not in the filtered items
    print("Generating dataset...")
    for chunk in tqdm(df_link_annotations, total=df_link_annotations_no_lines // CHUNKSIZE):
        for record in chunk.itertuples():
            if record.page_id not in all_pages:
                print(f"page_id: {record.page_id} not found in pages data")
                continue

            record_item_id = df_page_map.loc[record.page_id, 'item_id']

            if record_item_id not in all_qids:
                continue

            for section in record.sections:
                links = dict()

                for link_ind, link_page_id in enumerate(section["target_page_ids"]):
                    if link_page_id not in all_pages:
                        print(f"page_id: {link_page_id} not found in pages data")
                        continue

                    link_item_id = df_page_map.loc[link_page_id, 'item_id']

                    if link_item_id in all_qids:
                        links[
                            (
                                section["link_offsets"][link_ind],
                                section["link_offsets"][link_ind] + section["link_lengths"][link_ind],
                            )
                        ] = {link_item_id: 1.0}

                if links:
                   dataset.append((section["text"], {"links": links}))

    DATASET_PATH = DATASET_FOLDER_PATH / 'dataset.pkl' 
    print(f"Saving generated dataset to {DATASET_PATH}")
    pickle.dump(dataset, open(DATASET_PATH, "wb"))

if __name__ == '__main__':
    main()
  