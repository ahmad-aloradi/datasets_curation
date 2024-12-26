"""

"""

import yaml
import os
import pandas as pd
import numpy as np
import soundfile as sf
import logging
import stat
from time import strftime, gmtime
from sys import platform
from jiwer import compute_measures

data = []

DATASET_LOOKUP_TABLE = {
    "librispeech": "librispeech",
    "vctk-corpus": "vctk_corpus",
    "voxceleb": "voxceleb",
    "sitw": "sitw",
    "vpc2025": "vpc2025",
     }


def get_empty_dataframe():
    """Returns an empty dataframe with the relevant dataset fields

    Returns
    ------
    df: pandas dataframe
    """
    dtypes = np.dtype([
        ('dataset_name', str),
        ('annotation', str),
        ('phonemes', str),
        ('language', str),
        ('audio_file_type', str),
        ('samplerate', int),
        ('num_channels', int),
        ('speaker_id', str),
        ('gender', str),
        ('age', str),
        ('native', bool),
        ('country', str),
        ('dialect', str),
        ('recording_duration', float),
        ('comment', str),
        ('vad_start', float),
        ('vad_end', float),])
    data = np.empty(0, dtype=dtypes)
    df = pd.DataFrame(data)

    return df


def read_config(config_path):
    try:
        with open(config_path, 'rb') as yaml_file:
            return yaml.safe_load(yaml_file)
    except FileNotFoundError:
        raise FileNotFoundError('Config file {} does not exist'.format(config_path))


def get_audio_based_features(audio_filepath):
    audio = sf.SoundFile(audio_filepath)
    return pd.Series([audio.samplerate, 
                      audio.channels, 
                      str(audio.subtype), 
                      float(len(audio)/audio.samplerate)],
                      ['samplerate', 'num_channels', 'subtype', 'recording_duration'])


def write_dataset_csv(df, dataset_name, output_dir, delimiter='|', verbose=True, save_with_index=False):
    full_file_path = os.path.join(output_dir, dataset_name + '.csv')
    if not save_with_index:
        df.to_csv(full_file_path, encoding='utf-8', index=False, sep=delimiter)
    else:
        df.to_csv(full_file_path, encoding='utf-8', sep=delimiter)
    os.chmod(full_file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR | stat.S_IRGRP | stat.S_IWGRP | stat.S_IXGRP | stat.S_IROTH | stat.S_IWOTH | stat.S_IXOTH)

    if verbose:
        print('{}.csv successfully generated. See the directory {}'.format(dataset_name, output_dir))


def init_default_config_to_df(df, config):
    for key in config['common']:
        if key not in ['dataset_dir', 'audio_dir', 'annotation_dir', 'annotation_filepath', 'speaker_filepath',
                       'parser_dir', 'parser_file_type']:
            df[key] = config['common'][key]
    return df


def list_diff(list1, list2):
    """Finds different elements between two lists

    Parameters
    ----------
    list1: list
        List of elements

    list2: list
        List of elements

    Returns
    -------
    List
        List of different elements
    """
    c = set(list1).union(set(list2))  # or c = set(list1) | set(list2)
    d = set(list1).intersection(set(list2))  # or d = set(list1) & set(list2)
    return list(c - d)


def init_logging(dataset_name):
    # create logs directory, if it does not exist.
    if not os.path.isdir('logs'):
        os.makedirs('logs')

    timepoint_entry = strftime("%Y-%m-%d-%H-%M-%S", gmtime()) if platform == 'win32'\
        else strftime("%Y-%m-%d %H:%M:%S", gmtime())

    logging.basicConfig(filename='logs/{}_{}.log'.format(dataset_name, format(timepoint_entry)),
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        level=logging.DEBUG)

    console = logging.StreamHandler()
    console.setLevel(logging.INFO)

    # formatter
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)

    # add the handler to the root logger
    logging.getLogger('').addHandler(console)


def compute_total_wer(ground_truth, prediction):
    """Returns the total error between the ground truth text and prediction text.

    Parameters
    ----------
    ground_truth: str
        The ground truth annotation from a dataset.

    prediction: str
        Predicted annotation by asr.

    Returns
    -------
    : int
        Total value of substitutions, deletions, and insertions
        in WER calculation.
    """

    measures = compute_measures(ground_truth, prediction)

    return measures['substitutions'] + measures['deletions'] + measures['insertions']
