# Feb 2024 version of h5 parser from here https://github.com/mikemanookin/MEA/blob/63846c1e7af29c7b147204ea689df350ffc22b12/database/parse_data.py
import h5py
import json
import numpy as np
from scipy.signal import butter, filtfilt
from typing import Tuple
import bin2py
import os, argparse
import re
from datetime import datetime, timedelta


# RW_BLOCKSIZE = 2000000
# TTL_THRESHOLD = -1000
TTL_CHANNEL = 0
def get_litke_triggers(bin_path, RW_BLOCKSIZE=2000000, TTL_THRESHOLD=-1000):
    epoch_starts = []
    epoch_ends = []
    with bin2py.PyBinFileReader(bin_path, chunk_samples=RW_BLOCKSIZE, is_row_major=True) as pbfr:
        array_id = pbfr.header.array_id
        n_samples = pbfr.length
        for start_idx in range(0, n_samples, RW_BLOCKSIZE):
            n_samples_to_get = min(RW_BLOCKSIZE, n_samples - start_idx)
            samples = pbfr.get_data_for_electrode(0, start_idx, n_samples_to_get)
            # Find the threshold crossings at the beginning and end of each epoch.
            below_threshold = (samples < TTL_THRESHOLD)
            above_threshold = np.logical_not(below_threshold)
            # Epoch starts.
            above_to_below_threshold = np.logical_and.reduce([
                above_threshold[:-1],
                below_threshold[1:]
            ])
            trigger_indices = np.argwhere(above_to_below_threshold) + start_idx
            epoch_starts.append(trigger_indices[:, 0])
            below_to_above_threshold = np.logical_and.reduce([
                below_threshold[:-1],
                above_threshold[1:]
            ])
            trigger_indices = np.argwhere(below_to_above_threshold) + start_idx
            epoch_ends.append(trigger_indices[:, 0])
    epoch_starts = np.concatenate(epoch_starts, axis=0)
    epoch_ends = np.concatenate(epoch_ends, axis=0)
    return epoch_starts, epoch_ends, array_id, n_samples

def butter_lowpass_filter(data, cutoff, fs, order=6):
    nyq = 0.5 * fs  # Nyquist Frequency
    normal_cutoff = cutoff / nyq
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def find_threshold_cross(x, threshold, direction):
    x_original = x[:-1]
    x_shift = x[1:]
    if direction > 0:
        index = np.argwhere((x_original < threshold) & (x_shift >= threshold)).ravel()
    else:
        index = np.argwhere((x_original >= threshold) & (x_shift < threshold)).ravel()
    return index

def find_peaks(x, direction):
    # Take the second derivative.
    d2x = np.diff((np.diff(x) > 0.0).astype(float))
    if direction > 0: # local max
        index = np.argwhere(d2x < 0.0).ravel()
    else: # local min
        index = np.argwhere(d2x > 0.0).ravel()
    index += 1
    return index, x[index]

def get_frame_times(frame_monitor, bin_rate: float=1000.0, sample_rate: float=10000.0):
    default_frame_rate = 59.994
    # Low-pass filter fMonitor = lowPassFilter(fMonitor,250,1/sampleRate)
    frame_monitor = butter_lowpass_filter(frame_monitor, 250, sample_rate, 6)
    # Normalize
    frame_monitor -= np.min(frame_monitor)
    frame_monitor /= np.max(frame_monitor)
    ups, _ = find_peaks(frame_monitor,1)
    downs, _ = find_peaks(frame_monitor,-1)
    d_ups = np.diff(ups/sample_rate*1000.0)
    d_downs = np.diff(downs/sample_rate*1000.0)
    # Find the flips that are too short.
    short_ups = np.argwhere(d_ups < 30.0).ravel() + 1
    short_downs = np.argwhere(d_downs < 30.0).ravel() + 1
    ups = np.delete(ups, short_ups)
    downs = np.delete(downs, short_downs)
    frame_times = ups
    frame_times = np.append(frame_times, downs)
    # frame_times = np.append(frame_times, find_threshold_cross(frame_monitor,0.5,1))
    # frame_times = np.append(frame_times, find_threshold_cross(frame_monitor,0.5,-1))
    frame_times = np.sort(frame_times).astype(float)
    frame_times -= np.min(frame_times)
    frame_times = np.ceil(frame_times*bin_rate/sample_rate)
    return frame_times

def get_frame_times_lightcrafter(frame_monitor, bin_rate: float=1000.0, sample_rate: float=10000.0):
    # f_monitor = frame_monitor.copy()
    frame_monitor -= np.min(frame_monitor)
    frame_monitor /= np.max(frame_monitor)
    frame_ups = find_threshold_cross(frame_monitor,0.15,1)
    frame_downs = find_threshold_cross(frame_monitor,0.15,-1)

    # Get the upward transitions.
    d_ups = np.diff(frame_ups)
    up_idx = np.argwhere(d_ups > 15).ravel() + 1
    ups = frame_ups[up_idx]
    ups = np.append(ups, [0.0])
    
    # Get the downward transitions.
    d_downs = np.diff(frame_downs)
    down_idx = np.argwhere(d_downs > 15).ravel()
    downs = frame_downs[down_idx] + 3.0*sample_rate/1000.0
    frame_times = np.append(ups, downs)
    frame_times = np.sort(frame_times).astype(float)
    frame_times = np.ceil(frame_times*bin_rate/sample_rate)
    return frame_times

def get_frame_times_from_syncs(sync, bin_rate: float=1000.0, sample_rate: float=10000.0):
    sync -= np.min(sync)
    sync /= np.max(sync)
    frame_ups = find_threshold_cross(sync,0.5,1)
    frame_ups = np.insert(frame_ups,0,0)
    frame_times = frame_ups[::4]
    frame_times = np.sort(frame_times).astype(float)
    frame_times = np.ceil(frame_times*bin_rate/sample_rate)
    return frame_times

def get_frame_times_from_pwm(pwm: np.ndarray, bin_rate: float=1000.0, sample_rate: float=10000.0):
    d_pwm = np.diff(pwm)
    d_pwm = np.insert(d_pwm,0,0)
    pk, _ = find_peaks(d_pwm, 1)
    th_idx = find_threshold_cross(d_pwm,0.01,1)
    d_th_idx = np.diff(th_idx).astype(float)
    good_idx = np.argwhere(d_th_idx > sample_rate/1000.0*3.9).ravel() + 1
    good_idx = np.insert(good_idx,0,0)
    if (len(th_idx) == 0) or (len(good_idx) == 0):
        return np.array([])
    frame_times = th_idx[good_idx]
    frame_times = np.insert(frame_times,0,0)
    frame_times = frame_times[::4]
    frame_times = np.sort(frame_times).astype(float)
    frame_times = np.ceil(frame_times*bin_rate/sample_rate)
    return frame_times

def dotnet_ticks_to_datetime(ticks):
    t = datetime(1, 1, 1) + timedelta(microseconds = int(ticks)//10)
    date_string = t.strftime("%m/%d/%Y %H:%M:%S:%f")
    # Get epoch start times in seconds.
    date_seconds = t.timestamp()
    return date_string, date_seconds


def descend_obj(obj,sep='\t'):
    """
    Iterate through groups in a HDF5 file and prints the groups and datasets names and datasets attributes
    """
    if type(obj) in [h5py._hl.group.Group, h5py._hl.files.File]:
        for key in obj.keys():
            print(sep,'-',key,':',obj[key])
            descend_obj(obj[key],sep=sep+'\t')
    elif type(obj) == h5py._hl.dataset.Dataset:
        for key in obj.attrs.keys():
            print(sep+'\t','-',key,':',obj.attrs[key])

def h5dump(path,group='/'):
    """
    print HDF5 file metadata

    group: you can give a specific group, defaults to the root group
    """
    with h5py.File(path,'r') as f:
         descend_obj(f[group])

def hdf5_to_json(input_hdf5_file, output_json_file):
    import sys
    sys.setrecursionlimit(2000)
    with h5py.File(input_hdf5_file, 'r') as hdf5_file:
        data_dict = recursive_hdf5_to_dict(hdf5_file)
        
    with open(output_json_file, 'w') as json_file:
        json.dump(data_dict, json_file, cls=NpEncoder)

def recursive_hdf5_to_dict(group):
    result = dict()
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            result[key] = recursive_hdf5_to_dict(item)
        elif isinstance(item, h5py.Dataset):
            result[key] = item[()]  # Convert dataset to NumPy array
    return result


class NpEncoder(json.JSONEncoder):
    """ Special JSON encoder for numpy types. 
    Source: https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, bytes):
            return obj.decode('UTF-8')
        if isinstance(obj, h5py._hl.base.Empty):
            return None
        return super(NpEncoder, self).default(obj)

class Symphony2Reader:
    def __init__(self, 
                 h5_path: str, 
                 out_path: str=None, 
                 mea_raw_data_path: str=None, 
                 stage_type: str='LightCrafter',
                 save_h5_path: bool=False, 
                 sample_rate: float=20000.0):
        self.file = None
        self.h5_path = h5_path
        self.json_path = out_path
        self.mea_raw_data_path = mea_raw_data_path
        self.stage_type = stage_type
        self.save_h5_path = save_h5_path
        self.metadata = None
        self.experiment = None
        self.sample_rate = sample_rate
        try:
            splits = h5_path.split('/')
            self.experiment_name = splits[-1].split('.')[0]
        except:
            self.experiment_name = None

    def read_write(self):
        """ Write out the metadata as a JSON file. """
        self.metadata = self.read_file()
        self.organize_metadata()
        if self.json_path is not None:
            self.write_json(self.experiment, out_path) # self.write_json(self.metadata, out_path)
            # If this is an MEA experiment, export the text file.
            if self.mea_raw_data_path is not None:
                self.export_json(out_path)

    # Read in the hdf5 file.
    def read_file(self):
        """ Read in the hdf5 file.
        Parameters:
            file_path (str): The path to the hdf5 file.
        Returns:
            metadata (dict): The metadata extracted from the H5 file.
        """
        self.file = h5py.File(self.h5_path, 'r')
        keys = list(self.file.keys())
        for key in keys:
            # Find the experiment level.
            if type(self.file[key]) is h5py._hl.group.Group:
                metadata = self.parse_file(key)
                return metadata
        # with h5py.File(file_path, 'r') as file:
        #     keys = list(file.keys())
        #     for key in keys:
        #         # Find the experiment level.
        #         if type(file[key]) is h5py._hl.group.Group:
        #             metadata = self.parse_file(file, key)
        #             return metadata

    # Write out the metadata as a JSON file.
    def write_json(self, metadata, out_path):
        """ Write out the metadata as a JSON file.
        Parameters:
            metadata (dict): The metadata to write out.
            out_path (str): The path to write the JSON file.
        """
        with open(out_path, 'w+') as outfile:
            json.dump(metadata, outfile, cls=NpEncoder)
    
    def organize_metadata(self):
        """ Organize the metadata according to the hierarchy. """
        # Organize by protocol.
        protocol_labels = list()
        for group in self.metadata['group']:
            for block in group['block']:
                p_label = block['protocolID']
                if p_label not in protocol_labels:
                    protocol_labels.append(p_label)

        # Sort the labels.
        protocol_labels = sorted(protocol_labels)
        protocol_labels = np.array(protocol_labels)
        protocols = list()
        for p_label in protocol_labels:
            protocol = dict()
            protocol['label'] = p_label
            protocol['group'] = list()
            protocols.append(protocol)

        for group in self.metadata['group']:
            group_protocols = list()
            for block in group['block']:
                p_label = block['protocolID']
                if p_label not in group_protocols:
                    group_protocols.append(p_label)
            # Sort the labels.
            group_protocols = np.array(sorted(group_protocols))
            for g_label in group_protocols:
                p_idx = np.where(protocol_labels == g_label)[0][0]
                new_group = dict()
                new_group['attributes'] = group['attributes']
                new_group['label'] = group['label']
                new_group['properties'] = group['properties']
                new_group['source'] = group['source']
                new_group['block'] = list()
                for block in group['block']:
                    p_label = block['protocolID']
                    if (self.mea_raw_data_path is not None) and ('dataFile' in block.keys()):
                        block['dataFile'] = block['dataFile'].replace('.bin', '/')
                    if p_label == g_label:
                        new_group['block'].append(block)
                protocols[p_idx]['group'].append(new_group)

        self.experiment = dict()
        self.experiment['protocol'] = protocols
        self.experiment['sources'] = self.metadata['sources']
        self.experiment['project'] = self.metadata['project']

    # Get the name of the Litke data file saved for the EpochBlock
    def get_data_file_from_block(self, block: dict):
        """
        Extracts the data file name from the block dictionary.
        Parameters:
            block: Block dictionary (type: dict)
            Returns: Data file name (type: str)
        """
        dataFile = block['dataFile'].split('/')[-2:-1]
        dataFile = dataFile[0]
        return dataFile
    
    def export_json(self, filepath):
        """
        Exports the metadata to a text file.
        Parameters:
            filepath: path to the json file.
        """
        all_files = list()
        file_nums = list()
        out_path = filepath.replace('.json','.txt')
        with open(out_path, 'w') as file:
            for protocol in self.experiment['protocol']:
                file.write(protocol['label'] + '\n')
                group_dict = dict()
                for group in protocol['group']:
                    if group['label'] not in group_dict.keys():
                        group_dict[group['label']] = list()
                    for block in group['block']:
                        group_dict[group['label']].append(self.get_data_file_from_block(block))
                    # Sort the file list.
                    group_dict[group['label']] = sorted(group_dict[group['label']])
                for key, value in group_dict.items():
                    file.write('  -' + key + '\n')
                    for v in value:
                        file.write('    ->' + v + '\n')
                        all_files.append(v)
                        match = re.search('(\d+)',v)
                        file_nums.append(int(match.group(0)))
                
            # Sort the files.
            all_files = sorted(all_files)
            file.write('\n')
            file.write('All files:\n')
            for f in all_files:
                file.write(f + ' ')
            # Find any missing files.
            file_nums = sorted(file_nums)
            missing = self.missing_elements(file_nums)
            if len(missing) > 0:
                file.write('\n')
                file.write('Missing files: ')
                for m in missing:
                    file.write(str(m) + ' ')
            else:
                file.write('\n\n')
                file.write('No missing files (' + str(file_nums[0]) + '-' + str(file_nums[-1]) + ').')

    def missing_elements(self, L):
        L = sorted(L)
        start, end = L[0], L[-1]
        return sorted(set(range(start, end + 1)).difference(L))

    def parse_attributes(self, obj) -> dict:
        """ Parse the attributes.
        Parameters:
            obj (h5py._hl.group.Group): The attributes group.
        Returns:
            attributes (dict): The parsed attributes dictionary.
        """
        attributes = dict()
        for key, value in obj.attrs.items():
            if isinstance(value, np.bytes_):
                value = value.decode('UTF-8')
            elif isinstance(obj, h5py._hl.base.Empty):
                value = None
            attributes[key] = value
        return attributes

    def parse_properties(self, obj) -> dict:
        """ Parse the properties. 
        Parameters:
            obj (h5py._hl.group.Group): The properties group.
        Returns:
            props (dict): The parsed properties dictionary.
        """
        props = dict()
        for key, value in obj['properties'].attrs.items():
            if isinstance(value, np.bytes_):
                value = value.decode('UTF-8')
            elif isinstance(obj, h5py._hl.base.Empty):
                value = None
            props[key] = value
        return props

    def combine_parameters(self, block_params, epoch_params) -> dict:
        parameters = dict()
        for key, value in block_params.items():
            parameters[key] = value
        for key, value in epoch_params.items():
            parameters[key] = value
        parameters = dict(sorted(parameters.items()))
        return parameters

    def parse_epoch(self, epoch, block_params: dict) -> dict:
        epoch_dict = dict()
        # Get the attributes.
        attributes = self.parse_attributes(epoch)
        epoch_dict['attributes'] = attributes
        if 'label' in attributes.keys():
            epoch_dict['label'] = attributes['label']
        # Get the properties.
        epoch_dict['properties'] = dict()
        epoch_dict['parameters'] = dict()
        if 'properties' in epoch.keys():
            properties = self.parse_attributes(epoch['properties'])
            epoch_dict['properties'].update(properties)
        # Get the epoch backgrounds.
        backgrounds = dict()
        for key, value in epoch['backgrounds'].items():
            key_name = key.split('-')[0]
            bg_attrs = self.parse_attributes(value)
            if 'dataConfigurationSpans' in value.keys():
                try: 
                    foo = self.parse_attributes(value['dataConfigurationSpans'][list(value['dataConfigurationSpans'].keys())[0]][key_name])
                    # Append the items to the background attributes.
                    for bg_key, bg_value in foo.items():
                        bg_attrs[bg_key] = bg_value
                        if bg_key not in epoch_dict['properties']:
                            epoch_dict['parameters'][bg_key] = bg_value
                except Exception as error: 
                    print('Error: ' + str(error))
                    continue
            backgrounds[key_name] = bg_attrs
        epoch_dict['backgrounds'] = backgrounds
        # Parse the protocol parameters.
        params = dict()
        for key, value in epoch['protocolParameters'].attrs.items():
            params[key] = value
        params = self.combine_parameters(block_params, params)
        epoch_dict['parameters'].update(params)
        # Parse the responses.
        if 'responses' in epoch.keys():
            responses, frame_times = self.parse_responses(epoch['responses'])
            epoch_dict['responses'] = responses
            epoch_dict['frameTimesMs'] = frame_times
        # Parse the stimuli.
        if 'stimuli' in epoch.keys():
            stimuli = self.parse_stimuli(epoch['stimuli'])
            epoch_dict['stimuli'] = stimuli
        return epoch_dict

    def get_reference_string(self, value):
        """ Get the full reference string within the HDF5 file. 
        Parameters:
            value: H5 object
        Returns:
            full_path: Full reference string (type: str)
        """
        return value.name
        # if isinstance(value, h5py._hl.dataset.Dataset):
        #     return str(self.file[value.parent]).split('"')[1]
        # elif isinstance(value, h5py._hl.group.Group):
        #     return str(self.file[value.ref]).split('"')[1]
        # else:
        #     return None
    
    def parse_stimuli(self, stimuli: h5py._hl.group.Group) -> dict:
        """ Parse the stimuli. 
        Parameters:
            stimuli (h5py._hl.group.Group): The stimuli group.
        Returns:
            stimuli_dict (dict): The parsed stimuli dictionary.
        """
        stimuli_dict = dict()
        for key, value in stimuli.items():
            key_name = key.split('-')[0]
            stimuli_dict[key_name] = self.parse_attributes(value)
            # Get the full reference string within the HDF5 file.
            if self.save_h5_path:
                full_path = self.get_reference_string(value)
                stimuli_dict[key_name]['h5path'] = full_path
        return stimuli_dict

    def parse_responses(self, responses: h5py._hl.group.Group) -> Tuple[dict, np.ndarray]:
        """ Parse the responses. 
        Parameters:
            responses (h5py._hl.group.Group): The responses group.
        Returns:
            response_dict (dict): The parsed responses dictionary.
            frame_times (np.ndarray): The frame times.
        """
        response_dict = dict()
        frame_times = None
        found_syncs = False
        for key, value in responses.items():
            key_name = key.split('-')[0]
            response_dict[key_name] = self.parse_attributes(value)
            # Get the full reference string within the HDF5 file.
            if self.save_h5_path:
                full_path = self.get_reference_string(value)
                response_dict[key_name]['h5path'] = full_path
            # Pull the frame monitor data.
            if ('Sync' in key) and ('data' in value.keys()):
                red_sync = value['data']['quantity']
                sample_rate = response_dict[key_name]['sampleRate']
                # frame_times = get_frame_times_from_syncs(red_sync, bin_rate=1000.0, sample_rate=sample_rate)
                frame_times = get_frame_times_from_pwm(red_sync, bin_rate=1000.0, sample_rate=sample_rate)
            if ('Frame' in key) and ('data' in value.keys()) and (not found_syncs):
                frame_monitor = value['data']['quantity']#[()]
                sample_rate = response_dict[key_name]['sampleRate']
                if self.stage_type == 'LightCrafter':
                    frame_times = get_frame_times_lightcrafter(frame_monitor, bin_rate=1000.0, sample_rate=sample_rate)
                else:
                    frame_times = get_frame_times(frame_monitor, bin_rate=1000.0, sample_rate=sample_rate)
        return response_dict, frame_times

    def parse_epoch_block(self, epoch_block: dict) -> dict:
        """ Parse an epoch block.
        Parameters:
            epoch_block (dict): The epoch block to parse.
        Returns:
            dict: The parsed epoch block.
        """
        # Ensure that the block contains epochs, otherwise return None.
        if 'epochs' not in epoch_block.keys():
            print('WARNING: No epochs found in block: ')
            return None
        elif len(epoch_block['epochs']) == 0:
            print('WARNING: No epochs found in block: ')
            return None
        
        block = dict()
        unrecorded_epochs = None
        litke_starts = list()
        litke_ends = list()
        # Get the properties.
        if 'properties' in epoch_block.keys():
            properties = self.parse_attributes(epoch_block['properties'])
            if 'dataFileName' in properties:
                f_name = properties['dataFileName']
                f_name = f_name.replace('\\','/')
                f_name = f_name.replace('.bin', '/')
                block['dataFile'] = f_name
                # print('Parsing file: ' + f_name)
                if self.mea_raw_data_path is not None:
                    f_path = os.path.join(self.mea_raw_data_path, self.experiment_name, f_name.split('/')[-2])
                    litke_starts, litke_ends, array_id, n_samples = get_litke_triggers(f_path)
                    # Check for unrecorded epochs.
                    if np.any(np.array(litke_starts) > n_samples):
                        unrecorded_epochs = np.argwhere(np.array(litke_starts) > n_samples).ravel()
                        print('WARNING: Unrecorded epochs found in file: ' + f_name)
                        for bad_idx in sorted(unrecorded_epochs, reverse=True):
                            del litke_starts[bad_idx]
                            del litke_ends[bad_idx]
        # Get the attributes.
        attributes = self.parse_attributes(epoch_block)
        block['attributes'] = attributes
        if 'protocolID' in attributes.keys():
            block['protocolID'] = attributes['protocolID']
        # Get the parameters.
        params = self.parse_attributes(epoch_block['protocolParameters'])
        # Parse the epoch blocks.
        epoch_list = list()
        frame_times = list()
        for key, value in epoch_block['epochs'].items():
            my_epoch = self.parse_epoch(value, params)
            epoch_list.append(my_epoch)
            frame_times.append(my_epoch['frameTimesMs'])
        # Get the epochs ordered by start time.
        epoch_starts = np.zeros(len(epoch_list))
        for count, epoch in enumerate(epoch_list):
            epoch_starts[count] = epoch['attributes']['startTimeDotNetDateTimeOffsetTicks']
        epoch_order = np.argsort(epoch_starts)
        epoch_list = [epoch_list[i] for i in epoch_order]
        frame_times = [frame_times[i] for i in epoch_order]
        epoch_starts = epoch_starts[epoch_order]

        # Convert the epoch starts to a datetime.
        start_seconds = np.zeros(len(epoch_starts))
        epoch_datetime = list()
        for ii in range(len(epoch_starts)):
            date_string, date_seconds = dotnet_ticks_to_datetime(epoch_starts[ii])
            epoch_datetime.append(date_string)
            epoch_list[ii]['datetime'] = date_string
            # Get epoch start times in seconds.
            start_seconds[ii] = date_seconds
            # t = datetime(1, 1, 1) + timedelta(microseconds = epoch_starts[ii]//10)
            # epoch_datetime.append(t.strftime("%m/%d/%Y %H:%M:%S:%f"))
            # epoch_list[ii]['datetime'] = t.strftime("%m/%d/%Y %H:%M:%S:%f")
            # # Get epoch start times in seconds.
            # start_seconds[ii] = t.timestamp()

        if self.mea_raw_data_path is not None:
            if len(litke_starts) == 0:
                print('WARNING: No Litke triggers found in file: ' + f_name)
                litke_starts = 46969 + np.floor((start_seconds - start_seconds[0])*self.sample_rate).astype(int)
                unrecorded_epochs = np.argwhere(np.array(litke_starts) > n_samples).ravel()
                for bad_idx in sorted(unrecorded_epochs, reverse=True):
                    del epoch_list[bad_idx]
                    del frame_times[bad_idx]
                    del litke_starts[bad_idx]
            # Calculate the number of samples between each epoch from the Symphony file.
            start_samples = litke_starts[0] + np.floor((start_seconds - start_seconds[0])*self.sample_rate).astype(int)
            d_samps = np.mean(np.diff(start_samples))
            n_epochs = np.min([len(litke_starts), len(epoch_starts)])
            dt = np.abs(start_samples[:n_epochs] - litke_starts[:n_epochs])
            if np.any(dt > 0.5*d_samps):
                print('WARNING: Found missing epochs start pulses in file: ' + f_name)
                litke_starts = start_samples[:n_epochs]
            if len(epoch_starts) > len(litke_starts):
                print('WARNING: More epochs than Litke triggers found in file: ' + f_name)
                unrecorded_epochs = np.arange(len(litke_starts), len(epoch_starts))
                for bad_idx in sorted(unrecorded_epochs, reverse=True):
                    del epoch_list[bad_idx]
                    del frame_times[bad_idx]
            block['epochStarts'] = litke_starts
            block['epochEnds'] = litke_ends
            block['array_id'] = array_id
            block['n_samples'] = n_samples
        
        block['epoch'] = epoch_list
        block['frameTimesMs'] = frame_times
        return block

    def parse_epoch_group(self, epoch_group: h5py._hl.group.Group) -> dict:
        """Parse the epoch group.
        Parameters:
            epoch_group: h5py._hl.group.Group
        Returns:
            group_dict: dict
        """
        group_dict = dict()
        # Get the attributes.
        attributes = self.parse_attributes(epoch_group)
        group_dict['attributes'] = attributes
        if 'label' in attributes.keys():
            group_dict['label'] = attributes['label']
            print('Parsing group: ' + attributes['label'])
        # Get the properties.
        props = dict()
        for key, value in epoch_group['properties'].attrs.items():
            props[key] = value
        group_dict['properties'] = props
        # Parse the epoch blocks.
        block_list = list()
        for key, value in epoch_group['epochBlocks'].items():
            this_block = self.parse_epoch_block(value)
            if this_block is not None:
                block_list.append(this_block)
        group_dict['block'] = block_list
        # Get the group source.
        group_src = epoch_group['source'].attrs
        src = dict()
        if 'label' in group_src.keys():
            src['label'] = group_src['label'].decode('UTF-8')
        if 'uuid' in group_src.keys():
            src['uuid'] = group_src['uuid'].decode('UTF-8')
        group_dict['source'] = src
        return group_dict

    def parse_source(self, source: h5py._hl.group.Group, parse_children: bool=True) -> dict:
        """ Parse the source group.
        Parameters:
            source (h5py._hl.group.Group): The source group.
        Returns:
            source_dict (dict): The source dictionary.
        """
        source_dict = dict()
        # attributes = source.attrs
        # attrs = dict()
        # for key in attributes.keys():
        #     attrs[key] = attributes[key]
        attrs = self.parse_attributes(source)
        source_dict['attributes'] = attrs
        if 'label' in attrs.keys():
            source_dict['label'] = attrs['label']
        # props = dict()
        # for key, value in source['properties'].attrs.items():
        #     props[key] = value
        # source_dict['properties'] = props
        source_dict['properties'] = self.parse_properties(source)

        # Check for notes.
        if 'notes' in source.keys():
            notes = source['notes']
            source_dict['notes'] = self.parse_notes(notes)
        else:
            source_dict['notes'] = list()
        
        if parse_children:
            source_list = list()
            for key, value in source['sources'].items():
                source_list.append(self.parse_source(value))
            source_dict['sources'] = source_list
        return source_dict

    def parse_notes(self, notes: h5py._hl.group.Group) -> dict:
        """ Parse the notes group.
        Parameters:
            notes (h5py._hl.group.Group): The notes group.
        Returns:
            notes_dict (dict): The notes dictionary.
        """
        notes_list = list()
        note_text = notes['text']
        time_ticks = notes['time']['ticks']
        time_offset = notes['time']['offsetHours']
        for ii in range(len(note_text)):
            note_dict = dict()
            note_dict['text'] = note_text[ii].decode('UTF-8')
            note_dict['time_ticks'] = time_ticks[ii]
            note_dict['time_offsetHours'] = time_offset[ii]
            date_string, _ = dotnet_ticks_to_datetime(time_ticks[ii])
            note_dict['datetime'] = date_string
            notes_list.append(note_dict)
        return notes_list
    
    def parse_experiment(self, experiment: h5py._hl.group.Group) -> dict:
        """ Parse the experiment group.
        Parameters:
            experiment (h5py._hl.group.Group): The experiment group.
        Returns:
            experiment_dict (dict): The parsed experiment group.
        """
        experiment_dict = dict()
        attributes = experiment.attrs
        attrs = dict()
        for key in attributes.keys():
            attrs[key] = attributes[key]
        experiment_dict['attributes'] = attrs
        if 'label' in attrs.keys():
            experiment_dict['label'] = attrs['label']
        # Get the properties.
        props = dict()
        for key, value in experiment['properties'].attrs.items():
            props[key] = value
        experiment_dict['properties'] = props

        # Check for notes.
        if 'notes' in experiment.keys():
            notes = experiment['notes']
            experiment_dict['notes'] = self.parse_notes(notes)
        else:
            experiment_dict['notes'] = list()

        source_list = list()
        for key, value in experiment['sources'].items():
            source_list.append(self.parse_source(value))
        experiment_dict['sources'] = source_list
        return experiment_dict

    def parse_file(self, exp_key) -> dict:
        """ Parse the file for the experiment metadata.
        Parameters:
            file (h5py.File): The hdf5 file to parse.
            exp_key (str): The key for the experiment.
        Returns:
            metadata (dict): The metadata for the experiment.
        """
        metadata = dict()
        sources = self.file[exp_key]['sources']
        # Get the top-level/purpose node.
        project_dict = self.parse_source(self.file[exp_key], parse_children=False)
        # attributes = self.parse_attributes(self.file[exp_key])
        # Get the experiment level.
        exp_list = list()
        group_list = list()
        for _, value in sources.items():
            experiment = value['experiment']
            exp_sources = experiment['sources']
            print('Parsing experiment sources.')
            for _, s_value in exp_sources.items():
                exp_list.append(self.parse_experiment(s_value))
            exp_groups = experiment['epochGroups']
            group_count = 0
            for _, g_value in exp_groups.items():
                print('Parsing group {}'.format(group_count+1) + ' of {}...'.format(len(exp_groups)))
                group_list.append(self.parse_epoch_group(g_value))
                group_count+=1
        metadata['sources'] = exp_list
        metadata['group'] = group_list
        metadata['project'] = [project_dict]
        return metadata

    def close(self):
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.file.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Parse H5 experiment files into JSON format.')
    parser.add_argument('h5_path', type=str, help='Path to the Symphony HDF5 file (e.g., /data/data/h5/20230607C.h5).')
    parser.add_argument('out_path', type=str, help='Output path for the JSON and TXT files (e.g., /data/data/metadata/json/20230607C.json).')
    parser.add_argument('-r','--raw_data_path', default=None, type=str, help='Path to raw data (MEA experiments only; e.g., /data/data/raw/).')
    parser.add_argument('-s', '--stage-type', default='LightCrafter', type=str, help='Stage type (e.g., LightCrafter or Microdisplay).')
    parser.add_argument('-p','--save_h5_path', action='store_true', help='Whether to extract the full path to the stimuli/responses.')

    args = parser.parse_args()

    h5_path = args.h5_path
    out_path = args.out_path
    raw_data_path = args.raw_data_path
    stage_type = args.stage_type
    save_h5_path = args.save_h5_path
    print(save_h5_path)

    print('Parsing {}.'.format(h5_path))
    print('Writing to {}.'.format(out_path))
    print('Raw data path: {}'.format(raw_data_path))

    # h5_path = '/data/data/h5/20230607C.h5'
    # out_path = '/data/data/metadata/json/20230607C.json'
    # raw_data_path = '/data/data/raw/'

    with Symphony2Reader(h5_path=h5_path, out_path=out_path, mea_raw_data_path=raw_data_path, stage_type=stage_type, save_h5_path=save_h5_path) as reader:
        reader.read_write()
    # with Symphony2Reader(h5_path=args.h5_path, out_path=args.out_path, mea_raw_data_path=args.raw_data_path) as reader:
    #     reader.read_write()


# Testbed.
# import h5py
# from parse_data import Symphony2Reader
# h5_path='/Users/michaelmanookin/Documents/Data/rawdata/MEA_Data/test3.h5'
# out_path = '/Users/michaelmanookin/Documents/Data/rawdata/MEA_Data/test3.json'
# stage_type='Lightcrafter'
# self = Symphony2Reader(h5_path=h5_path, out_path=out_path, stage_type=stage_type)
# self.file = h5py.File(self.h5_path, 'r')
# keys = list(self.file.keys())
# exp_key='experiment-8958ede8-3f3e-4aa8-a7d1-d3e1dbd9a653'
# sources = self.file[exp_key]['sources']
# responses = self.file['/experiment-632a688f-c321-4075-9d92-45e09d4b78b6/epochGroups/epochGroup-c81563a0-9938-493a-b8a8-83867772b24a/epochBlocks/edu.washington.riekelab.protocols.SingleSpot-18647ef5-577c-45b6-9d6e-9063c400cf75/epochs/epoch-225b1086-6c69-4fcd-b6f8-f1461421fe28/']