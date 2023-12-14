"""
General purpose class for:
Parsing spike outputs of a given protocol
Parsing receptive field parameters in .params and {algo}_params.mat
"""

import numpy as np
import os
import visionloader as vl
import hdf5storage
import pickle
import celltype_io as ctio
import symphony_data as sd
import matplotlib.pyplot as plt

class SpikeOutputs(object):
    def __init__(self, str_experiment, str_datafile=None, str_protocol=None, str_algo=None,
                 paramsfile=None, dataset_name=None, paramsmatfile=None,
                 str_classification=None, ls_RGC_labels=['OffP', 'OffM', 'OnP', 'OnM', 'SBC'],
                 str_chunk=None, ls_filenames=None):
        self.str_experiment = str_experiment
        self.str_datafile = str_datafile
        self.str_protocol = str_protocol
        self.str_algo = str_algo

        self.paramsfile = paramsfile
        self.dataset_name = dataset_name
        self.paramsmatfile = paramsmatfile
        self.str_chunk = str_chunk
        self.ls_filenames = ls_filenames
        self.ls_RGC_labels = ls_RGC_labels

        # Load classifications if provided
        if str_classification is not None:
            self.str_classification = str_classification
            self.types = ctio.CellTypes(str_classification, ls_RGC_labels=ls_RGC_labels)

        # Load datafile if provided
        if str_datafile is not None:
            ls_data_keys = ['spike_dict', 'isi', 'cluster_id', 'acf',
                            'isi_bin_edges']
            if '.mat' in str_datafile:
                self.stim = hdf5storage.loadmat(str_datafile)
            elif '.p' in str_datafile:
                with open(str_datafile, 'rb') as f:
                    self.stim = pickle.load(f)
                    self.spikes = {}

                    # Populate data dict with keys from ls_data_keys if present
                    for key in ls_data_keys:
                        if key in self.stim.keys():
                            self.spikes[key] = self.stim[key]
                            
                            # Remove from stim
                            self.stim.pop(key)

            else:
                raise ValueError('Data file must be .mat or .p')
            self.ARR_CELL_IDS = np.array(self.spikes['cluster_id']).flatten()
            self.GOOD_CELL_IDS = self.ARR_CELL_IDS.copy()
            self.N_CELLS = len(self.ARR_CELL_IDS)

    def exclude_isi_violations(self, n_bin_max=4, refractory_threshold=0.1):
        # acf has 0.5ms bins, n_bin_max is index of bins to count spikes upto. 4 means first four bins will be included.
        # Calculate the percentage of spikes that violate refractoriness (<2ms isi)
        acf = self.spikes['acf']
        pct_refractory = np.sum(acf[:,:n_bin_max], axis=1) * 100
        # May have to play with the cutoff
        idx_bad = np.argwhere((pct_refractory > refractory_threshold))[:,0]
        
        # Remove bad IDs from ARR_CELL_IDs
        self.pct_refractory = pct_refractory
        self.bad_isi_idx = idx_bad
        self.bad_isi_ids = self.ARR_CELL_IDS[idx_bad]
        self.good_isi_idx = np.delete(np.arange(self.N_CELLS), idx_bad)
        self.GOOD_CELL_IDS = np.delete(self.ARR_CELL_IDS, idx_bad)
        self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)

        print(f'Of {self.N_CELLS} cells, {len(idx_bad)} cells removed and {self.N_GOOD_CELLS} cells remaining.')
    
    def load_sta(self, paramsfile: str=None, dataset_name: str=None, paramsmatfile: str=None):
        if not paramsfile:
            paramsfile = self.paramsfile
            dataset_name = self.dataset_name
            paramsmatfile = self.paramsmatfile
        self.vcd = vl.load_vision_data(analysis_path=os.path.dirname(paramsfile), dataset_name=dataset_name, include_params=True,
                                       include_runtimemovie_params=True, include_ei=True)
        self.N_WIDTH = self.vcd.runtimemovie_params.width
        self.N_HEIGHT = self.vcd.runtimemovie_params.height
        
        # This is used to translate noise stixel space to microns.
        self.NOISE_GRID_SIZE = self.vcd.runtimemovie_params.micronsPerStixelX # Typically 30 microns. 
        
        # Load RF fit parameters from .params file
        self.d_sta = self.vcd.main_datatable
        sta_cell_ids = list(self.d_sta.keys())

        # Load _params.mat. 
        if paramsmatfile:
            self.d_params = hdf5storage.loadmat(paramsmatfile)
                
            # Get spatial maps of cells
            self.d_sta_spatial = {}
            for idx_ID, n_ID in enumerate(sta_cell_ids):
                # Load red channel spatial map. Cell ID index in vcd should be same as in _params.mat
                self.d_sta_spatial[n_ID] = self.d_params['spatial_maps'][idx_ID, :, :, 0]
            
            # Get convex hull fits of cells
            self.d_sta_convex_hull = {}
            for idx_ID, n_ID in enumerate(sta_cell_ids):
                self.d_sta_convex_hull[n_ID] = self.d_params['hull_vertices'][idx_ID, :,:]
                

        if 'noise' in self.str_protocol.lower():
            self.ARR_CELL_IDS = np.array(sta_cell_ids)
            self.GOOD_CELL_IDS = np.array(sta_cell_ids)
            self.N_CELLS = len(self.ARR_CELL_IDS)
            self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)
        # else:
        #     # Get cell IDs that appear in both protocol and STA
        #     self.ARR_COMMON_CELL_IDS = np.intersect1d(self.GOOD_CELL_IDS, sta_cell_ids)
        #     self.N_COMMON_CELLS = self.ARR_COMMON_CELL_IDS.shape[0]

    def filter_low_nspikes_noise(self, n_percentile=10):
        # Filter cells with low number of spikes in noise protocol for each cell type
        print('Filtering low noise nspikes cells...')
        for idx_t, str_type in enumerate(self.ls_RGC_labels):
            type_IDs = self.types.d_main_IDs[str_type]
            
            arr_nSpikes = np.array([self.vcd.main_datatable[str_ID]['EI'].n_spikes for str_ID in type_IDs])
            n_cutoff = np.percentile(arr_nSpikes, n_percentile)
            arr_keep = arr_nSpikes > n_cutoff

            # Update type IDs
            self.types.d_main_IDs[str_type] = type_IDs[arr_keep]
            print(f'{arr_keep.sum()} / {len(type_IDs)} {str_type} cells are kept.')

            # Update good cell IDs by deleting type_IDs not kept
            self.GOOD_CELL_IDS = np.delete(self.GOOD_CELL_IDS, 
                                           np.argwhere(np.isin(self.GOOD_CELL_IDS, type_IDs[~arr_keep])))
        
        self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)
        print('Done.')


    def filter_low_crf_f1(self, crf_datafile, n_percentile=10, contrast=0.05):
        # Filter cells with low F1 amplitude for specified contrast
        print('Filtering low CRF F1 cells...')
        # Load crf datafile
        with open(crf_datafile, 'rb') as f:
            d_crf = pickle.load(f)
        crf_ids = d_crf['cluster_id']
        contrast_vals = d_crf['unique_params']['contrast']
        idx_contrast = np.argwhere(contrast_vals == contrast)[0][0]

        # Get F1 amplitudes for each cell type
        for idx_t, str_type in enumerate(self.ls_RGC_labels):
            type_IDs = self.types.d_main_IDs[str_type]
            
            # arr_nSpikes = np.array([self.vcd.main_datatable[str_ID]['EI'].n_spikes for str_ID in type_IDs])
            type_idx = ctio.map_ids_to_idx(type_IDs, crf_ids)
            arr_f1 = d_crf['4Hz_amp'][type_idx, idx_contrast]
            
            n_cutoff = np.percentile(arr_f1, n_percentile)
            arr_keep = arr_f1 > n_cutoff

            # Update type IDs
            self.types.d_main_IDs[str_type] = type_IDs[arr_keep]
            print(f'{arr_keep.sum()} / {len(type_IDs)} {str_type} cells are kept.')

            # Update good cell IDs by deleting type_IDs not kept
            self.GOOD_CELL_IDS = np.delete(self.GOOD_CELL_IDS, 
                                           np.argwhere(np.isin(self.GOOD_CELL_IDS, type_IDs[~arr_keep])))
        
        self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)
        self.d_crf = d_crf
        print('Done.')


    def load_psth(self, str_protocol, ls_param_names, bin_rate=100.0):
        c_data = sd.Dataset(self.str_experiment)
        self.param_names = ls_param_names
        self.str_protocol = str_protocol
        self.bin_rate = bin_rate

        spike_dict, cluster_id, params, unique_params, pre_pts, stim_pts, tail_pts = c_data.get_spike_rate_and_parameters(
        str_protocol, None, self.param_names, sort_algorithm=self.str_algo, bin_rate=bin_rate, sample_rate=20000,
        file_name=self.ls_filenames)

        n_epochs = spike_dict[cluster_id[0]].shape[0]
        n_bin_dt = 1/bin_rate * 1000 # in ms
        
        # Check that pre_pts, stim_pts, tail_pts all have a uniform value
        for pts in [pre_pts, stim_pts, tail_pts]:
            if len(np.unique(pts)) != 1:
                raise ValueError(f'{pts} has more than one unique value.')
            
        n_pre_pts = int(pre_pts[0])
        n_stim_pts = int(stim_pts[0])
        n_tail_pts = int(tail_pts[0])
        
        n_total_pts = n_pre_pts + n_stim_pts + n_tail_pts

        self.stim = {'params': params, 'unique_params': unique_params, 
             'n_epochs': n_epochs, 'n_pre_pts': n_pre_pts, 'n_stim_pts': n_stim_pts, 'n_tail_pts': n_tail_pts,
             'n_total_pts': n_total_pts, 'bin_rate': bin_rate, 'n_bin_dt': n_bin_dt}
        self.spikes = {'spike_dict': spike_dict, 'cluster_id': cluster_id}
        self.ARR_CELL_IDS = np.array(cluster_id)
        self.GOOD_CELL_IDS = np.array(cluster_id)
        self.N_CELLS = len(self.ARR_CELL_IDS)
        self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)

    def load_isi(self, str_protocol, bin_edges=np.linspace(0,300,601)):
        # Get ISI
        c_data = sd.Dataset(self.str_experiment)
        acf, isi, isi_cluster_id = c_data.get_interspike_interval(str_protocol, None, self.str_algo, 
                                                     file_name=self.ls_filenames, bin_edges=bin_edges)
        self.spikes['isi'] = isi
        self.spikes['isi_cluster_id'] = isi_cluster_id
        self.spikes['acf'] = acf
        self.spikes['isi_bin_edges'] = bin_edges
        

    def set_common_ids(self):
        self.sta_cluster_ids = list(self.d_sta.keys())
        self.ARR_COMMON_CELL_IDS = np.intersect1d(self.GOOD_CELL_IDS, self.spikes['cluster_id'])
        self.ARR_COMMON_CELL_IDS = np.intersect1d(self.ARR_COMMON_CELL_IDS, self.sta_cluster_ids)
        self.N_COMMON_CELLS = self.ARR_COMMON_CELL_IDS.shape[0]
        print('Number of common cells: ' + str(self.N_COMMON_CELLS))

        # For each of main types, set to common cells
        self.types.d_main_IDs_original = self.types.d_main_IDs.copy()
        for str_type in self.types.d_main_IDs.keys():
            arr_common_ids = np.intersect1d(self.types.d_main_IDs[str_type], self.ARR_COMMON_CELL_IDS)
            print(f'Out of {len(self.types.d_main_IDs[str_type])} {str_type} cells, {len(arr_common_ids)} are common.')
            self.types.d_main_IDs[str_type] = arr_common_ids

    def print_stim_summary(self):
        # Print stim summary from stim dictionary
        print('epoch length: ' + str(self.stim['n_total_pts'] * self.stim['n_bin_dt']) + ' ms')
        print('Total epochs: ' + str(self.stim['n_epochs']))
        print('pre: ' + str(self.stim['n_pre_pts'] * self.stim['n_bin_dt']) + ' ms; stim: ' + str(self.stim['n_stim_pts'] * self.stim['n_bin_dt']) + ' ms; tail: ' + str(self.stim['n_tail_pts'] * self.stim['n_bin_dt']) + ' ms')
        print('pre pts: ' + str(self.stim['n_pre_pts']) + '; stim pts: ' + str(self.stim['n_stim_pts']) + '; tail pts: ' + str(self.stim['n_tail_pts']))
        print('bin rate: ' + str(self.stim['bin_rate']) + ' Hz; bin dt: ' + str(self.stim['n_bin_dt']) + ' ms')
        
    def save(self, str_path: str=None):
        if not str_path:
            str_path = os.path.join(self.str_experiment, self.str_datafile)
        d_save = {'stim': self.stim, 'spikes': self.spikes}
        with open(str_path, 'wb') as f:
            pickle.dump(d_save, f)
        print('Saved to ' + str_path)

    def load(self, str_path: str):
        with open(str_path, 'rb') as f:
            d_load = pickle.load(f)
        self.stim = d_load['stim']
        self.spikes = d_load['spikes']
        cluster_id = d_load['spikes']['cluster_id']
        self.ARR_CELL_IDS = np.array(cluster_id)
        self.GOOD_CELL_IDS = np.array(cluster_id)
        self.N_CELLS = len(self.ARR_CELL_IDS)
        self.N_GOOD_CELLS = len(self.GOOD_CELL_IDS)
        print('Loaded from ' + str_path)

    def remove_dups(self, thresh, str_type):
        type_IDs = self.types.d_main_IDs[str_type]
        n_type_cells = len(type_IDs)

        arr_rfs = [(self.d_sta[str_ID]['x0'], self.d_sta[str_ID]['y0']) for str_ID in type_IDs]

        # Copy of RFs
        # Loop through copy. While loop through original. If distance < thresh, remove from original.

        # Compute pairwise distance between all RFs
        arr_dist = np.zeros((n_type_cells, n_type_cells))
        # Get triu indices
        # triu_idx = np.triu_indices(n_type_cells, k=1) This gives some bugs
        triu_idx = (range(n_type_cells), range(n_type_cells))
        for i in triu_idx[0]:
            for j in triu_idx[1]:
                arr_dist[i, j] = np.sqrt((arr_rfs[i][0] - arr_rfs[j][0])**2 + (arr_rfs[i][1] - arr_rfs[j][1])**2)


        # Remove IDs with distance < thresh, keeping only the first one
        arr_keep = np.ones(n_type_cells, dtype=bool)
        for i in triu_idx[0]:
            for j in triu_idx[1]:
                if arr_dist[i, j] < thresh and i != j:
                    arr_keep[j] = False

        arr_keep_IDs = type_IDs[arr_keep]

        # Print summary
        print(f'Number of {str_type} cells: {len(arr_keep_IDs)} of {len(type_IDs)}')

        return arr_keep_IDs

       
        