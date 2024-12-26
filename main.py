import os
from argparse import ArgumentParser
from utils import (DATASET_LOOKUP_TABLE, 
                   read_config,
                   get_empty_dataframe,
)
from datasets_process import librispeech

def run(dataset_name, delimiter='|'):
    """Runs the generation of the .csv file of the corresponding dataset

    Parameters
    ----------
    dataset_name: str
        Name of the dataset. For the available datasets, see the README.md

    delimiter: string
        Delimiter for the .csv file.
    """
    df = get_empty_dataframe()
    config_path = os.path.join('config', dataset_name + '.yaml')
    config = read_config(config_path)

    try:
        dataset_lookup = DATASET_LOOKUP_TABLE[dataset_name]
    except KeyError:
        raise KeyError('Dataset is not available')

    globals()[dataset_lookup].process(df, config, delimiter)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-d', '--dataset_name',
                        type=str,
                        help='Dataset name of which .csv file to be generated.',
                        metavar='<DatasetName>')

    parser.add_argument('-D', '--delimiter',
                        type=str,
                        default='|',
                        help='Delimiter to split the fields in the .csv file.',
                        metavar='<Delimiter>')

    # Extract arguments from the argument parser
    args = parser.parse_args()
    dataset_name = args.dataset_name
    delimiter = args.delimiter
    run(dataset_name.lower(), delimiter)
