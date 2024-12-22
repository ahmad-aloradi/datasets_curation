"""
LibriSpeech dataset preparation
"""

import os
import glob
import pandas as pd
from multiprocessing.dummy import Pool as ThreadPool
from tqdm import tqdm

from utils import get_audio_based_features, init_default_config_to_df, write_dataset_csv


def process(df, config, delimiter):
    """Generates .csv file for the dataset

    Parameters
    ----------
    df: pandas dataframe
        For the schema, see utils.get_empty_dataframe()

    config: dictionary
        Read from the corresponding .yaml config file.

    delimiter: string
        Delimiter for the .csv file.
    """

    # Read the speaker text into a dataframe
    df_speaker = pd.read_table(os.path.join(config['common']['dataset_dir'], config['common']['speaker_filepath']),
                               sep='|', index_col=False,
                               names=['ID', 'SEX', 'SUBSET', 'MINUTES', 'NAME'],
                               skiprows=lambda x: x in range(0, 12)).drop(['MINUTES', 'NAME', 'SUBSET'], axis=1)
    # Rename columns
    df_speaker = df_speaker.rename(columns={"ID": "speaker_id", "SEX": "gender"})

    # M -> male & F-> female
    df_speaker['gender'] = df_speaker['gender'].apply(lambda gender: 'male' if 'M' in gender else 'female')

    # speaker_id must be a string
    df_speaker['speaker_id'] = df_speaker['speaker_id'].apply(lambda speaker_id: str(speaker_id))

    # Get the list of relative audio filepaths from directory
    audio_rel_filepaths_from_dir = get_audio_rel_filepaths_from_dir(config)
    text_files = [text_file for text_file in glob.glob(os.path.join(config['common']['dataset_dir'], '*/*/*/*.txt')) if
                  os.path.isfile(text_file)]

    annotations = list()
    speaker_ids = list()
    audio_rel_filepaths = list()

    for text_file in tqdm(text_files, total=len(text_files)):
        text_file_rel_path = os.path.relpath(text_file, config['common']['dataset_dir'])
        subfolder, speaker_id, _, _ = text_file_rel_path.split('/')
        audio_rel_dir = os.path.dirname(text_file_rel_path)
        with open(text_file) as f:
            for line in f:
                audio_filename, annotation = line.split(' ', 1)
                # Omit new line at the end of annotations
                annotation = annotation.strip('\n')
                audio_rel_filepath = os.path.join(audio_rel_dir, audio_filename + '.' +
                                                  config['common']['audio_file_type'])

                audio_rel_filepaths.append(audio_rel_filepath)
                speaker_ids.append(str(speaker_id))
                annotations.append(annotation)

    assert len(audio_rel_filepaths_from_dir) == len(annotations) == len(audio_rel_filepaths), \
        'Audio files and annotations don\'t match'

    # Write the information extracted within the loop into the dataframe
    df['audio_rel_filepath'] = audio_rel_filepaths
    df['annotation'] = annotations
    df['speaker_id'] = speaker_ids

    # Initialize the common parameters (config['common']) into the dataframe
    init_default_config_to_df(df, config)

    df = df.join(df_speaker.set_index('speaker_id'), on='speaker_id', rsuffix='_')
    df['gender'] = df['gender_']
    df = df.drop(['gender_'], axis=1)

    df['speaker_id'] = df['speaker_id'].apply(lambda s_id: 'librispeech_' + s_id)

    # Add audio based features to the df (this may take a while)
    print('Processing audio based features...')
    for subset in [df['common']['train_dir'], df['common']['test_dir'], df['common']['dev_dir']]:
        print('Processing subset: {}'.format(subset))
        with ThreadPool(100) as p:
            df[['samplerate', 'num_channels', 'recording_duration']] =\
                p.map(get_audio_based_features, df['audio_rel_filepath'].
                    apply(lambda rel_filepath: os.path.join(subset, rel_filepath)))

    # Add database name as prefix to avoid any future ambiguity due to non-unique speaker_ids
    df['speaker_id'] = df['speaker_id'].apply(lambda speaker_id: 'librispeech_' + str(speaker_id))

    # Write the dataframe
    #   with the name of the dataset
    #   to the base directory of the dataset
    #   in .csv format.
    write_dataset_csv(df=df,
                      dataset_name=config['common']['dataset_name'],
                      output_dir=config['common']['dataset_dir'],
                      delimiter=delimiter)


def get_audio_rel_filepaths_from_dir(config):
    # Find all the audio and text files in the dataset directory
    audio_filepaths = [audio_filepath for audio_filepath in glob.glob(
        os.path.join(config['common']['dataset_dir'], '*/*/*/*.{}'.format(config['common']['audio_file_type']))) if
                   os.path.isfile(audio_filepath)]
    print('Total speech files found in the audio directory: {}'.format(len(audio_filepaths)))

    audio_rel_filepaths_from_dir = [os.path.relpath(audio_filepath, config['common']['dataset_dir'])
                                    for audio_filepath in audio_filepaths]

    return audio_rel_filepaths_from_dir
