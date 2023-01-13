import argparse
import pickle
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--filtered_items', type=str, required=True, help='Path to file with set of filtered items [pickle format]')
    parser.add_argument('--link_annotations', type=str, required=True, help='Path to link annotated text file [json format]')
    parser.add_argument('--page_map', type=str, required=True, help='Path to page to item map file [csv format]')
    args = parser.parse_args()

    FILTERED_ITEMS_PATH = Path(args.filtered_items)
    LINK_ANNOTATIONS_PATH = Path(args.link_annotations)
    PAGE_MAP_PATH = Path(args.page_map)
    DATASET_FOLDER_PATH = (Path(__file__).parents[1] / 'data' / 'dataset')
    
    # Create path if not exists
    DATASET_FOLDER_PATH.mkdir(parents=True, exist_ok=True)

    CHUNKSIZE = 1000

    # Load filtered items
    print("Loading files...")
    filtered_items = pickle.load(open(FILTERED_ITEMS_PATH, "rb"))
    df_link_annotations = pd.read_json(LINK_ANNOTATIONS_PATH, lines=True, chunksize=CHUNKSIZE)
    df_page_map = pd.read_csv(PAGE_MAP_PATH, index_col=0)

    # Get size of df_link_annotations
    df_link_annotations_no_lines = sum(1 for line in open(LINK_ANNOTATIONS_PATH))

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

            if record_item_id not in filtered_items:
                continue

            for section in record.sections:
                links = dict()

                for link_ind, link_page_id in enumerate(section["target_page_ids"]):
                    if link_page_id not in all_pages:
                        print(f"page_id: {link_page_id} not found in pages data")
                        continue

                    link_item_id = df_page_map.loc[link_page_id, 'item_id']

                    if link_item_id in filtered_items:
                        links[
                            (
                                section["link_offsets"][link_ind],
                                section["link_offsets"][link_ind] + section["link_lengths"][link_ind],
                            )
                        ] = {link_item_id: 1.0}

                if links:
                   dataset.append((section["text"], {"links": links}))

    DATASET_PATH = DATASET_FOLDER_PATH / f'{time.strftime("%Y-%m-%dT%H%M%S")}_dataset.pkl' 
    print(f"Saving generated dataset to {DATASET_PATH}")
    pickle.dump(dataset, open(DATASET_PATH, "wb"))

if __name__ == '__main__':
    main()
  