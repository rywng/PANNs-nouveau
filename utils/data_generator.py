import logging
import random
import os

import librosa
import numpy as np
import polars as pl
from torch.utils.data import Dataset


class AudioSetDatasetCsv(Dataset):
    def __init__(self, classes_num, sample_rate=32000, audio_len_sec=10):
        """This class takes path of audio as input, and return
        the waveform and target of the audio clip. This class is used by DataLoader.
        """
        self.sample_rate = sample_rate
        self.audio_len_sec = audio_len_sec
        self.classes_num = classes_num

    def pad_or_sample(self, input: np.ndarray):
        target_frame_len = self.sample_rate * self.audio_len_sec
        if input.shape[0] < target_frame_len:
            return np.concatenate(
                (input, np.zeros(target_frame_len - input.shape[0])), axis=0
            )
        else:
            start_pos = random.randint(0, len(input) - target_frame_len)
            return input[start_pos : start_pos + target_frame_len]

    def one_hot_encode(self, input, start=1) -> np.ndarray:
        res = []
        for i in range(self.classes_num):
            res.append(1 if input == i else 0)
        return np.array(res, dtype=np.float_)

    def __getitem__(self, df_row: dict):
        """Load waveform and target of an audio clip.

        Args:
          df_row: {
            path: str
            label: int
          }

        Returns:
          data_dict: {
            'audio_name': str,
            'waveform': (clip_samples,),
            'target': (classes_num,)}
        """
        audio_path = df_row["path"]
        audio_name = os.path.basename(audio_path)
        waveform, sample_rate = librosa.load(audio_path, sr=self.sample_rate)
        assert (
            sample_rate == self.sample_rate
        ), f"Loaded waveform sample rate {sample_rate} differs from target rate {self.sample_rate}"

        waveform = self.pad_or_sample(waveform)

        data_dict = {
            "audio_name": audio_name,
            "waveform": waveform,
            "target": self.one_hot_encode(df_row["label"]),
        }  # target: onehot encoded label

        return data_dict


class CsvBase(object):
    def __init__(self, csv_path: str, batch_size: int, random_seed=None):
        """Base class of csv-based sampler.

        Args:
            csv_path: string
            batch_size: int
            random_seed: int
        """
        self.batch_size = batch_size
        self.random_state = np.random.RandomState(random_seed)
        self.dataframe = pl.read_csv(csv_path)
        self.num_labels = self.dataframe.n_unique("label")
        if self.batch_size % self.num_labels != 0:
            logging.warn(
                f"batch_size {self.batch_size} is not appropriate for {self.num_labels} of labels, which might introduce data unbalance"
            )
        # Random sample the dataframe
        self.dataframe = self.dataframe.sample(
            fraction=1.0, seed=random_seed, shuffle=True
        )
        # label dicts: {label: data_df}
        self.label_dicts = {
            label: data_df for label, data_df in self.dataframe.group_by("label")
        }


class CsvTrainSampler(CsvBase):
    def __init__(self, csv_path: str, batch_size: int, random_seed=None):
        super(CsvTrainSampler, self).__init__(csv_path, batch_size, random_seed)

    def __iter__(self):
        # find the longest length k for label a,b....n , return
        # [data[a][i % len(a)], data[b][i % len(b)] .... data[n][i%len(n)]] for i in range(k)
        max_len = max(value.shape[0] for value in self.label_dicts.values())

        i = 0
        INCREMENT_STEP = (
            self.batch_size // self.num_labels
        )  # should increment i by batch_size // num_labels every time, to avoid duplication

        while i in range(max_len - INCREMENT_STEP + 1):
            res_batch = []
            for label in self.label_dicts:
                for item in (
                    self.label_dicts[label]
                    .slice(
                        i % self.label_dicts[label].shape[0],
                        INCREMENT_STEP,
                    )
                    .to_dicts()
                ):
                    res_batch.append(item)

            i += INCREMENT_STEP
            yield res_batch


class EvaluateSampler(object):
    def __init__(self, indexes_hdf5_path, batch_size):
        """Evaluate sampler. Generate batch meta for evaluation.

        Args:
          indexes_hdf5_path: string
          batch_size: int
        """
        self.batch_size = batch_size

        with h5py.File(indexes_hdf5_path, "r") as hf:
            self.audio_names = [
                audio_name.decode() for audio_name in hf["audio_name"][:]
            ]
            self.hdf5_paths = [hdf5_path.decode() for hdf5_path in hf["hdf5_path"][:]]
            self.indexes_in_hdf5 = hf["index_in_hdf5"][:]
            self.targets = hf["target"][:].astype(np.float32)

        self.audios_num = len(self.audio_names)

    def __iter__(self):
        """Generate batch meta for training.

        Returns:
          batch_meta: e.g.: [
            {'hdf5_path': string,
             'index_in_hdf5': int}
            ...]
        """
        batch_size = self.batch_size
        pointer = 0

        while pointer < self.audios_num:
            batch_indexes = np.arange(
                pointer, min(pointer + batch_size, self.audios_num)
            )

            batch_meta = []

            for index in batch_indexes:
                batch_meta.append(
                    {
                        "audio_name": self.audio_names[index],
                        "hdf5_path": self.hdf5_paths[index],
                        "index_in_hdf5": self.indexes_in_hdf5[index],
                        "target": self.targets[index],
                    }
                )

            pointer += batch_size
            yield batch_meta


def collate_fn(list_data_dict):
    """Collate data.
    Args:
      list_data_dict, e.g., [{'audio_name': str, 'waveform': (clip_samples,), ...},
                             {'audio_name': str, 'waveform': (clip_samples,), ...},
                             ...]
    Returns:
      np_data_dict, dict, e.g.,
          {'audio_name': (batch_size,), 'waveform': (batch_size, clip_samples), ...}
    """
    np_data_dict = {}

    for key in list_data_dict[0].keys():
        np_data_dict[key] = np.array([data_dict[key] for data_dict in list_data_dict])

    return np_data_dict
