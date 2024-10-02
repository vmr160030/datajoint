"""
Functions and classes for QC of MEA data.
Main methods:
ISI rejection
NumSpikes rejection
CRF rejection
EI correlation
"""

import numpy as np
import spikeoutputs as so
import spikeplots as sp
import celltype_io as ctio
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def remove_cells_by_ids(data: so.SpikeOutputs, ids):
    """Remove cells from SpikeOutputs object by ids
    Args:
        data: SpikeOutputs object
        ids: list of cell ids to remove
    Returns:
        data: SpikeOutputs object with cells removed
    """
    # Remove bad IDs from GOOD_CELL_IDS
    print(f'Removing {len(ids)}/{data.N_CELLS} = {len(ids)/data.N_CELLS:.2f} cells.')

    idx_bad = np.where(np.isin(data.ARR_CELL_IDS, ids))[0]
    good_ids = np.delete(data.ARR_CELL_IDS, idx_bad)
    data.GOOD_CELL_IDS = np.intersect1d(good_ids, data.GOOD_CELL_IDS)
    data.N_GOOD_CELLS = len(data.GOOD_CELL_IDS)
    
    return data

def exclude_isi_violations(data: so.SpikeOutputs, str_protocol: str=None, n_bin_max=4, refractory_threshold=0.1):
    # acf has 0.5ms bins, n_bin_max is index of bins to count spikes upto. 4 means first four bins will be included.
    # Calculate the percentage of spikes that violate refractoriness (<2ms isi)
    if str_protocol is not None:
        acf = data.isi[str_protocol]['acf']
    else:
        acf = data.spikes['acf']
    pct_refractory = np.sum(acf[:,:n_bin_max], axis=1) * 100
    # May have to play with the cutoff
    idx_bad = np.argwhere((pct_refractory > refractory_threshold))[:,0]
    
    # Remove bad IDs from ARR_CELL_IDs
    data.pct_refractory = pct_refractory
    data.bad_isi_idx = idx_bad
    data.bad_isi_ids = data.ARR_CELL_IDS[idx_bad]
    data.good_isi_idx = np.delete(np.arange(data.N_CELLS), idx_bad)

    data = remove_cells_by_ids(data, data.bad_isi_ids)
    print(f'{len(data.bad_isi_ids)} cells removed due to ISI violations.')

def filter_low_nspikes_noise(data: so.SpikeOutputs, n_percentile=10):
    # Filter cells with low number of spikes in noise protocol for each cell type
    print('Filtering low noise nspikes cells...')
    for idx_t, str_type in enumerate(data.ls_RGC_labels):
        type_IDs = data.types.d_main_IDs[str_type]
        
        arr_nSpikes = np.array([len(data.vcd.main_datatable[n_ID]['SpikeTimes']) for n_ID in type_IDs])
        n_cutoff = np.percentile(arr_nSpikes, n_percentile)
        arr_keep = arr_nSpikes > n_cutoff

        # Update type IDs
        data.types.d_main_IDs[str_type] = type_IDs[arr_keep]
        print(f'{arr_keep.sum()} / {len(type_IDs)} {str_type} cells are kept.')

        # Update good cell IDs by deleting type_IDs not kept
        data.GOOD_CELL_IDS = np.delete(data.GOOD_CELL_IDS, 
                                        np.argwhere(np.isin(data.GOOD_CELL_IDS, type_IDs[~arr_keep])))
    
    data.N_GOOD_CELLS = len(data.GOOD_CELL_IDS)
    print('Done.')

def filter_low_crf_f1(data: so.SpikeOutputs, crf_datafile, n_percentile=10, contrast=0.05):
    # Filter cells with low F1 amplitude for specified contrast
    print('Filtering low CRF F1 cells...')
    # Load crf datafile
    with open(crf_datafile, 'rb') as f:
        d_crf = pickle.load(f)
    crf_ids = d_crf['cluster_id']
    contrast_vals = d_crf['unique_params']['contrast']
    idx_contrast = np.argwhere(contrast_vals == contrast)[0][0]

    # Get F1 amplitudes for each cell type
    for idx_t, str_type in enumerate(data.ls_RGC_labels):
        type_IDs = data.types.d_main_IDs[str_type]
        
        # arr_nSpikes = np.array([self.vcd.main_datatable[str_ID]['EI'].n_spikes for str_ID in type_IDs])
        type_idx = ctio.map_ids_to_idx(type_IDs, crf_ids)
        arr_f1 = d_crf['4Hz_amp'][type_idx, idx_contrast]
        
        n_cutoff = np.percentile(arr_f1, n_percentile)
        arr_keep = arr_f1 > n_cutoff

        # Update type IDs
        data.types.d_main_IDs[str_type] = type_IDs[arr_keep]
        print(f'{arr_keep.sum()} / {len(type_IDs)} {str_type} cells are kept.')

        # Update good cell IDs by deleting type_IDs not kept
        data.GOOD_CELL_IDS = np.delete(data.GOOD_CELL_IDS, 
                                        np.argwhere(np.isin(data.GOOD_CELL_IDS, type_IDs[~arr_keep])))
    
    data.N_GOOD_CELLS = len(data.GOOD_CELL_IDS)
    data.d_crf = d_crf
    print('Done.')

def remove_dups(data: so.SpikeOutputs, thresh, str_type, b_update=True, b_plot=True,
                sd_mult=1):
    type_IDs = data.types.d_main_IDs[str_type]
    n_type_cells = len(type_IDs)

    rfs = [(data.d_sta[str_ID]['x0']*data.NOISE_GRID_SIZE, 
            data.d_sta[str_ID]['y0']*data.NOISE_GRID_SIZE) for str_ID in type_IDs]

    # Compute pairwise distance between all RFs
    arr_dist = np.zeros((n_type_cells, n_type_cells))
    arr_dist[:] = np.inf

    dist_idx = (range(n_type_cells), range(n_type_cells))
    for i in dist_idx[0]:
        for j in dist_idx[1]:
            if i != j:
                arr_dist[i, j] = np.sqrt((rfs[i][0] - rfs[j][0])**2 + (rfs[i][1] - rfs[j][1])**2)

    # Copy of RFs and dist
    dedup_idx = np.arange(n_type_cells)
    dedup_dist = arr_dist.copy()
    
    # While loop through copy. If distance < thresh, remove from copy.
    while np.any(dedup_dist < thresh):
        # Find min dist
        min_idx = np.unravel_index(np.argmin(dedup_dist), dedup_dist.shape)
        
        # Remove row and col from copy
        dedup_dist = np.delete(dedup_dist, min_idx[0], axis=0)
        dedup_dist = np.delete(dedup_dist, min_idx[1], axis=1)
        
        # Remove from idx
        dedup_idx = np.delete(dedup_idx, min_idx[0])

    # Save deduped IDs
    dedup_id = np.array([type_IDs[i] for i in dedup_idx])
    data.types.d_main_IDs[str_type+'_dd'] = dedup_id

    # Plot
    if b_plot:
        import spikeplots as sp
        f, axs = plt.subplots(ncols=2, figsize=(10,5))
        sp.plot_type_rfs(data, [str_type, str_type+'_dd'], axs=axs,
        b_zoom=True, sd_mult=sd_mult)
        # del self.types.d_main_IDs[str_type+'_dd']

    # Update data
    if b_update:
        data.types.d_main_IDs[str_type+'_all'] = type_IDs
        data.types.d_main_IDs[str_type] = dedup_id

    
    return dedup_dist, dedup_id

def find_dup_thresh(data: so.SpikeOutputs, str_type: str, arr_thresh=np.arange(20,100,5)):
    # For each threshold, plot number of cells remaining
    type_IDs = data.types.d_main_IDs[str_type]
    ls_n_cells = []
    for thresh in arr_thresh:
        _, dedup_id = remove_dups(data, thresh, str_type, b_update=False, b_plot=False)
        ls_n_cells.append(len(dedup_id))
    f, ax = plt.subplots()
    ax.plot(arr_thresh, ls_n_cells)


class QC(object):
    def __init__(self, data: so.SpikeOutputs, b_noise_only: bool=False):
        self.data = data
        self.ls_RGC_labels = data.ls_RGC_labels
        self.N_CELLS = data.N_CELLS
        self.ARR_CELL_IDS = data.ARR_CELL_IDS
        self.GOOD_CELL_IDS = data.GOOD_CELL_IDS
        self.N_GOOD_CELLS = data.N_GOOD_CELLS

        ls_cols = ['cell_id', 'cell_type', 'noise_spikes',
                    'noise_isi_violations', 'crf_f1', 'ei_corr']
        if not b_noise_only:
            ls_cols += ['protocol_spikes', 'protocol_isi_violations']
        self.df_qc = pd.DataFrame(columns=ls_cols)
        
        # Populate df_qc cell_id and cell_type
        self.df_qc['cell_id'] = self.ARR_CELL_IDS

        # Set cell_id as index
        self.df_qc.set_index('cell_id', inplace=True)

        # Get IDs for noise and protocol
        self.noise_ids = np.array(list(data.vcd.main_datatable.keys()))
        

        # Populate total spike counts
        ls_noisespikes = []
        ls_protocolspikes = []
        ls_celltypes = []
        for n_ID in self.ARR_CELL_IDS:
            # Check if n_ID in vcd
            try:   #RACHEL added try/except to handle cells in VCD without SpikeTimes key
                if n_ID in data.vcd.main_datatable.keys():
                    ls_noisespikes.append(len(data.vcd.main_datatable[n_ID]['SpikeTimes']))
                else:
                    ls_noisespikes.append(0)
            except KeyError:
                ls_noisespikes.append(0)
            # Check if n_ID in data.types.d_types
            if n_ID in data.types.d_types.keys():
                ls_celltypes.append(data.types.d_types[n_ID])
            else:
                ls_celltypes.append('not in typing.txt')
        
        self.df_qc['cell_type'] = ls_celltypes
        # Dict for threshold sets
        self.d_thresh = {'set1': {'df_keep': self.df_qc.copy()}}
        self.df_qc['noise_spikes'] = ls_noisespikes
       
        
        # populate noise ISI violations
        n_refractory_period = 1.5 # ms
        self.n_refractory_period = n_refractory_period
        isi_bin_edges = data.isi[data.str_noise_protocol]['isi_bin_edges']
        isi_bin_edges = np.array([(isi_bin_edges[i], isi_bin_edges[i+1]) for i in range(len(isi_bin_edges)-1)])
        n_bin_max = np.argwhere(isi_bin_edges[:,1] <= n_refractory_period)[-1][0] + 1
        print(f'Using first {n_bin_max} bins for refractory period calculation.')
        print(f'isi_bin_edges: {isi_bin_edges[:n_bin_max]}')
        noise_isi = data.isi[data.str_noise_protocol]['acf']
        pct_refractory = np.sum(noise_isi[:,:n_bin_max], axis=1) * 100
        self.pct_refractory = pct_refractory
        
        ids_noise_isi = data.isi[data.str_noise_protocol]['isi_cluster_id']
        self.df_qc.loc[ids_noise_isi, 'noise_isi_violations'] = pct_refractory

        # Get protocol data
        if not b_noise_only:
            self.protocol_ids = np.array(data.spikes['cluster_id'])
            for n_ID in data.ARR_CELL_IDS:
                # Check if n_ID in data.spikes['cluster_id']
                if n_ID in data.spikes['cluster_id']:
                    n_idx = np.argwhere(data.spikes['cluster_id'] == n_ID)[0][0]
                    ls_protocolspikes.append(data.spikes['total_spike_counts'][n_idx])
                else:
                    ls_protocolspikes.append(0)
            self.df_qc['protocol_spikes'] = ls_protocolspikes
            
            # populate protocol ISI violations
            protocol_isi = data.isi[data.str_protocol]['acf']
            pct_refractory = np.sum(protocol_isi[:,:n_bin_max], axis=1) * 100
            self.df_qc.loc[self.protocol_ids, 'protocol_isi_violations'] = pct_refractory


    def set_abs_thresh(self, str_set, str_param, n_thresh, b_keep_below=True):
        # Set threshold for str_param in str_set
        d_thresh_vals = {'n_thresh': n_thresh, 'b_keep_below': b_keep_below, 'b_by_type': False}

        # Apply absolute threshold to df_keep
        df_qc = self.df_qc
        df_keep = self.d_thresh[str_set]['df_keep']
        if b_keep_below:
            df_keep[str_param] = df_qc[str_param] < n_thresh
        else:
            df_keep[str_param] = df_qc[str_param] > n_thresh
        d_thresh_vals['n_cells'] = df_keep[str_param].sum()

        for str_type in self.data.ls_RGC_labels:
            n_total = df_keep[df_keep['cell_type'] == str_type].shape[0]
            d_thresh_vals[f'n_{str_type}'] = df_keep[df_keep['cell_type'] == str_type][str_param].sum()
            d_thresh_vals[f'pct_{str_type}'] = d_thresh_vals[f'n_{str_type}'] / n_total

        self.d_thresh[str_set][str_param] = d_thresh_vals
    
    def set_pct_thresh_by_type(self, str_set, str_param, n_top_pct):
        # Set threshold for str_param in str_set
        d_thresh_vals = {'n_top_pct': n_top_pct, 'b_by_type': True}

        # Apply percentile threshold to df_keep
        df_qc = self.df_qc
        df_keep = self.d_thresh[str_set]['df_keep']

        for str_type in self.data.ls_RGC_labels:
            df_qc_slice = df_qc[df_qc['cell_type'] == str_type]
            n_total = df_qc_slice.shape[0]
            slice_idx = df_qc_slice.index
            arr_param = df_qc_slice[str_param]
            n_cutoff = np.percentile(arr_param, 100-n_top_pct)

            df_keep.loc[slice_idx, str_param] = arr_param > n_cutoff
            d_thresh_vals[str_type] = n_cutoff
            d_thresh_vals[f'n_{str_type}'] = df_keep.loc[slice_idx, str_param].sum()
            d_thresh_vals[f'pct_{str_type}'] = d_thresh_vals[f'n_{str_type}'] / n_total
        
        # NaN to True in df_keep
        df_keep[str_param].fillna(True, inplace=True)
        self.d_thresh[str_set][str_param] = d_thresh_vals
    
    def plot_dist_by_type(self, str_param, ax=None, b_plot_thresh=False, str_set='set1'):
        if ax is None:
            f, ax = plt.subplots()
        df_plot = self.df_qc[self.df_qc['cell_type'].isin(self.data.ls_RGC_labels)]
        sns.boxplot(x='cell_type', y=str_param, data=df_plot, ax=ax,
            order=self.data.ls_RGC_labels)
        

        if b_plot_thresh and str_set in self.d_thresh.keys():
            # Plot threshold
            d_vals = self.d_thresh[str_set][str_param]
            
            if d_vals['b_by_type']:
                for idx_t, str_type in enumerate(self.data.ls_RGC_labels):
                    ax.axhline(d_vals[str_type], color=f'C{idx_t}', linestyle='--')
                    ax.text(idx_t, d_vals[str_type], f'{d_vals[str_type]:.2f}', 
                        color='k', fontsize=8, ha='center', va='bottom')
                    # ax.text(idx_t, d_vals[str_type], f'{pct_cells:.2f}', 
                    #     color='k', fontsize=8, ha='center', va='bottom')
                    
            else:
                ax.axhline(d_vals['n_thresh'], color='k', linestyle='--')
                ax.set_title(f'{str_param} threshold: {d_vals["n_thresh"]}')        
        return ax
    
    def plot_pct_by_type(self, str_param, ax=None, str_set='set1'):
        if ax is None:
            f, ax = plt.subplots()

        d_vals = self.d_thresh[str_set][str_param]
        pct = np.array([d_vals[f'pct_{str_type}'] for str_type in self.ls_RGC_labels])
        df_pct = pd.DataFrame({'cell_type': self.ls_RGC_labels, 'pct': pct})
        ax=sns.barplot(x='cell_type', y='pct', data=df_pct, ax=ax, order=self.ls_RGC_labels)
        ax.bar_label(ax.containers[0], fmt='%.2f')
        ax.set_ylabel('Percent cells kept')
        if d_vals['b_by_type']:
            ax.set_title(f'{str_param} top {d_vals["n_top_pct"]} percentile')
        else:
            ax.set_title(f'{str_param} threshold: {d_vals["n_thresh"]}')
        return ax
    
    def plot_ncells_by_type(self, str_param, ax=None, str_set='set1'):
        if ax is None:
            f, ax = plt.subplots()

        d_vals = self.d_thresh[str_set][str_param]
        n_cells = np.array([d_vals[f'n_{str_type}'] for str_type in self.ls_RGC_labels])
        df_ncells = pd.DataFrame({'cell_type': self.ls_RGC_labels, 'n_cells': n_cells})
        ax=sns.barplot(x='cell_type', y='n_cells', data=df_ncells, ax=ax, order=self.ls_RGC_labels)
        ax.bar_label(ax.containers[0], fmt='%.0f')
        ax.set_ylabel('Number of cells kept')
        if d_vals['b_by_type']:
            ax.set_title(f'{str_param} top {d_vals["n_top_pct"]} percentile')
        else:
            ax.set_title(f'{str_param} threshold: {d_vals["n_thresh"]}')
        return ax
    
    
    def plot_report(self, str_param, str_set='set1'):
        f, axs = plt.subplots(nrows=2, ncols=1, figsize=(10,10))
        self.plot_dist_by_type(str_param, ax=axs[0], b_plot_thresh=True, str_set=str_set)
        # self.plot_pct_by_type(str_param, ax=axs[1], str_set=str_set)
        self.plot_ncells_by_type(str_param, ax=axs[1], str_set=str_set)

        return f, axs
    
    def get_intersection_cells(self, str_set='set1'):
        # Get cell IDs (index of df_keep) that are True for all parameters
        df_keep = self.d_thresh[str_set]['df_keep']
        df_keep = df_keep.select_dtypes(include='bool')
        arr_keep = df_keep.values

        arr_intersection = np.all(arr_keep, axis=1)
        return df_keep.index[arr_intersection].values


    