import pathlib
import argparse
import csv


dataset_path = pathlib.Path(__file__).parent.parent.parent / 'dataset' / 'dataset.csv'


def get_args():
    parser = argparse.ArgumentParser(description='Update the dataset')
    parser.add_argument('--path', type=str, default=dataset_path.absolute(), help='Path to the dataset')
    parser.add_argument('--rows', type=list, nargs='+', help='Rows to add to the dataset')


def update_dataset(path, rows):
    with open(path, 'a') as f:
        writer = csv.writer(f)
        writer.writerows(rows)

if __name__ == '__main__':
    args = get_args()
    update_dataset(args.path, args.rows)