import bin2py
import numpy as np
import matplotlib.pyplot as plt
import sys
import preproc_present_images as ppi
import symphony_data as sd
# sys.path.append('/Users/riekelabbackup/Desktop/Vyom/MEA/database/')
# import parse_data as pd

def load_epoch_starts(exp_name: str, str_protocol: str, file_names: list):
    c_data = sd.Dataset(exp_name)
    d_p = c_data.M.search_data_file(str_protocol, file_name=file_names)
    epoch_starts = []
    for d_g in d_p['group']:
        for d_b in d_g['block']:
            es = d_b['epochStarts']
            if isinstance(es, list):
                # If es is a list, it contains multiple epoch starts
                for e in es:
                    epoch_starts.append(e)
            else:
                # If es is a single value, append it directly
                epoch_starts.append(es)

    return epoch_starts

# Constants
RW_BLOCKSIZE = 100000  # Block size for reading data
TTL_THRESHOLD = 1000
class RawTraces:
    def __init__(self, binpath, exp_name: str, str_protocol: str, file_names: list, d_time: dict):
        self.binpath = binpath
        self.data = None
        self.ttl_times = None
        self.ttl_samples = None
        self.sample_rate = 20000  # Hz
        # print(f'Loading epoch starts, ends, array_id, and n_samples from {self.binpath}')
        # epoch_starts, epoch_ends, array_id, n_samples = pd.get_litke_triggers(self.binpath)
        # self.epoch_ends = epoch_ends
        # self.array_id = array_id
        # self.n_samples = n_samples
        # print(f"Epoch ends: {self.epoch_ends}")
        # print(f"Array ID: {self.array_id}")
        # print(f"Number of samples: {self.n_samples}")

        print(f'Loading epoch starts from {exp_name} with files {file_names}')
        self.epoch_starts = load_epoch_starts(exp_name, str_protocol, file_names)
        print(f"Epoch starts with # trials: {len(self.epoch_starts)}")
        ls_samples_per_epoch = []
        for i in range(len(d_time['preTime'])):
            trial_time = d_time['preTime'][i] + d_time['stimTime'][i] + d_time['tailTime'][i]

            trial_samples = int((trial_time/1000) * self.sample_rate)
            ls_samples_per_epoch.append(trial_samples)

        self.samples_per_epoch = np.array(ls_samples_per_epoch)
        print(f'Average samples per epoch: {np.mean(self.samples_per_epoch)} = {np.mean(self.samples_per_epoch)/self.sample_rate} seconds')

    def load_bin_data(self, start_sample=0, end_sample=None, verbose=False):
        """
        Load raw .bin data into a NumPy array.

        Parameters:
            binpath (str): Path to the .bin file.
            start_sample (int): Starting sample index (default: 0).
            end_sample (int): Ending sample index (default: None, reads till the end).

        Returns:
            np.ndarray: Loaded data as a NumPy array of shape [electrodes, samples].
        """
        with bin2py.PyBinFileReader(self.binpath, chunk_samples=RW_BLOCKSIZE, is_row_major=True) as pbfr:
            # Determine the number of electrodes and total samples
            n_channels = pbfr.num_electrodes
            total_samples = pbfr.length
            SAMPLE_RATE = 20000 # Hz
            if verbose:
                print(f"Number of electrodes: {n_channels}, Total samples: {total_samples}.")
                print(f"Total time: {total_samples / SAMPLE_RATE} seconds")
                print(f"Sample rate: {SAMPLE_RATE} Hz")

            # Set end_sample to the total length if not specified
            if end_sample is None:
                end_sample = total_samples

            # Validate sample range
            if start_sample < 0 or end_sample > total_samples or start_sample >= end_sample:
                raise ValueError("Invalid start_sample or end_sample range.")

            query_samples = end_sample - start_sample
            if verbose:
                print(f"Querying {query_samples} samples from {start_sample} to {end_sample}.")
                print(f'Queried time: {query_samples / SAMPLE_RATE} seconds')
                print(f'From {start_sample / SAMPLE_RATE} to {end_sample / SAMPLE_RATE} seconds')
                
            # Preallocate array for the data
            data = np.zeros((n_channels, query_samples), dtype=np.float32)

            ttl_times_buffer = []
            ttl_samples = np.zeros((query_samples,), dtype=np.float32)
            # Read data in chunks
            for start_idx in range(start_sample, end_sample, RW_BLOCKSIZE):
                n_samples_to_get = min(RW_BLOCKSIZE, end_sample - start_idx)
                chunk = pbfr.get_data(start_idx, n_samples_to_get)

                # Extract TTL data (channel 0) and compute TTL times
                ttl_samples = chunk[0, :]
                below_threshold = (ttl_samples < -TTL_THRESHOLD)
                above_threshold = np.logical_not(below_threshold)
                below_to_above = np.logical_and.reduce([
                    below_threshold[:-1],
                    above_threshold[1:]
                ])
                trigger_indices = np.argwhere(below_to_above) + start_idx
                ttl_times_buffer.append(trigger_indices[:, 0])

                # Populate the data matrix (exclude channel 0)
                data[:, start_idx - start_sample:start_idx - start_sample + n_samples_to_get] = chunk[1:, :]
                # ttl_samples[start_idx - start_sample:start_idx - start_sample + n_samples_to_get] = chunk[0, :]

            # Concatenate TTL times
            ttl_times = np.concatenate(ttl_times_buffer, axis=0)
        
        if verbose:
            print(f'Data shape: {data.shape}')
        # print(f'TTL times shape: {ttl_times.shape}')
        self.data = data
        self.ttl_times = ttl_times
        self.ttl_samples = ttl_samples
        
    def load_epoch_index(self, epoch_idx):
        self.load_bin_data(start_sample=self.epoch_starts[epoch_idx],
                           end_sample=self.epoch_starts[epoch_idx] + self.samples_per_epoch[epoch_idx])


def load_epoch_data(str_bin_path, epoch_idx):
    epoch_starts, epoch_ends, array_id, n_samples = pd.get_litke_triggers(str_bin_path)

    # Get the start and end of the specified epoch
    epoch_start = epoch_starts[epoch_idx]
    epoch_end = epoch_ends[epoch_idx]

    # Load the data for the specified epoch
    raw_data, ttl_times, ttl_samples = load_bin_data(str_bin_path, start_sample=epoch_start, end_sample=epoch_end)

    d_output = {
        'raw_data': raw_data,
        'ttl_times': ttl_times,
        'ttl_samples': ttl_samples,
        'epoch_start': epoch_start,
        'epoch_end': epoch_end,
        'array_id': array_id
    }
    return d_output

def get_image_flash_onsets(data, all_fts):
    d_bins = ppi.get_transition_bins_by_state_time(data, frame_rate=59, stride=1,
                                               pre_flash_time=0.25,verbose=True)

    # Get stimulus image IDs, (n_epochs, n_imgs_per_epoch)
    img_strs = np.array([str_imgs.split(',') for str_imgs in data.stim['params']['imageName']])
    # Remove .png and convert to int
    img_ids = np.zeros(img_strs.shape, dtype=int)
    for i in range(len(img_strs)):
        for j in range(len(img_strs[i])):
            img_ids[i, j] = int(img_strs[i, j].split('.')[0])

    # Get img onsets
    img_onset_frames = np.zeros(img_ids.shape, dtype=int)
    img_onset_samples = np.zeros(img_ids.shape, dtype=int)
    # For every epoch
    for i in range(img_ids.shape[0]):
        pre_frames = d_bins['pre_bins']
        img_onset_frames[i, 0] = pre_frames
        # For every image in the epoch
        for j in range(img_ids.shape[1]-1):
            # Get flash + gap frames of current image
            n_frames_for_img = d_bins['ls_flash_bins'][j] + d_bins['ls_gap_bins'][j]
            img_onset_frames[i, j+1] = img_onset_frames[i, j] + n_frames_for_img

    lcr_frame_rate = 59.941548817817917
    for i in range(img_onset_frames.shape[0]):
        # e_fts = all_fts[i]
        for j in range(img_onset_frames.shape[1]):
            # img_onset_samples[i, j] = e_fts[img_onset_frames[i, j]]
            img_onset_samples[i, j] = int(np.round(img_onset_frames[i, j] / lcr_frame_rate * 20000))

    return img_onset_samples, img_onset_frames, img_ids