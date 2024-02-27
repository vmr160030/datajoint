"""
CRF protocol data extraction and analysis.

Primary output is '4Hz_amp' which is array of shape [n_clusters, n_contrast_levels]
Values are the peak-trough amplitude of the 4Hz component of the avg PSTH for 1 cycle of CRF.
This is equivalent to 2*amp of a 4Hz sine wave fit to the PSTH.

Clusters are indexed by 'cluster_id'

'mean_psth_cycle' is array of shape [n_clusters, n_bins_per_cycle, n_contrast_levels]

Optionally, entire PSTH timecourse can be saved with --savefull True
'avg_psth' is array of shape [n_clusters, n_bins, n_contrast_levels]
This will increase output file size a lot.

Usage:
    python crf_analysis.py 20221216C -f data009 -a kilosort2
"""

import numpy as np
from scipy.fftpack import fft
import argparse
from symphony_data import Dataset, Analysis
import pickle
import config as cfg
import os
import pickle
import dj_metadata as djm
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
import spikeplots as sp

# Compute 4Hz amp
def compute_4Hz_amp(arr_psth_cycle: np.ndarray):
    """Computes peak-trough amplitude of 4Hz component of PSTH.

    Args:
        arr_psth_cycle (np.ndarray): 1D array of 1 cell's avg PSTH for 1 cycle of CRF.

    Returns:
        _type_: _description_
    """
    n_bins_per_cycle = arr_psth_cycle.shape[0]
    arr_ft = np.abs(fft(arr_psth_cycle)[:n_bins_per_cycle//2]) # Magnitude of positive frequencies
    arr_ft = 2/n_bins_per_cycle*arr_ft # Normalize by number of bins, x2 for pos and neg frequencies
    return arr_ft[1] * 2 # x2 for peak-trough amp

def fetch_data(str_experiment, str_search='ContrastResponseGrating', str_group=None, str_algorithm='kilosort2',
               bin_rate=1000.0, sample_rate=20000.0, ls_fnames=None, b_savefull=False):
    print('Loading data...')
    # Get the avg spike response for each stimulus condition.
    d = Dataset(str_experiment)

    param_names = ['contrast', 'barWidth', 'temporalFrequency', 'orientation', 
                   'apertureClass', 'spatialClass', 'temporalClass', 'chromaticClass']
    
    spike_dict, cluster_id, params, unique_params, pre_pts, stim_pts, tail_pts = d.get_spike_rate_and_parameters(
        str_search, str_group, param_names, sort_algorithm=str_algorithm, bin_rate=bin_rate,
        sample_rate=sample_rate, file_name=ls_fnames)

    
    if len(unique_params['temporalFrequency']) > 1:
        raise ValueError('More than one temporal frequency in data.')
    else:
        print('Temporal frequency: ' + str(unique_params['temporalFrequency'][0]) + ' Hz')

    # Compute avg PSTH of single cycle
    n_bins_per_cycle = int(bin_rate / params['temporalFrequency'][0])

    n_unique_contrasts = len(np.unique(params['contrast']))
    arr_mean_psth_cycle = np.zeros((len(cluster_id), n_bins_per_cycle, n_unique_contrasts), dtype=np.float32)
    avg_psth = np.zeros((len(cluster_id), spike_dict[cluster_id[0]].shape[1], n_unique_contrasts), dtype=np.float32)
    
    print('Computing average PSTH of single cycle for {} contrasts'.format(n_unique_contrasts))
    for i, c in enumerate(cluster_id):
        for j, contrast in enumerate(np.unique(params['contrast'])):
            arr_psth = np.mean(spike_dict[c][params['contrast'] == contrast, :], axis=0)
            avg_psth[i, :, j] = arr_psth
            arr_mean_psth_cycle[i, :, j] = arr_psth.reshape((-1, n_bins_per_cycle)).mean(axis=0)
    
    arr_4Hz_amp = np.zeros((len(cluster_id), n_unique_contrasts))
    for cell_idx in range(len(cluster_id)):
        for contrast_idx in range(n_unique_contrasts):
            arr_4Hz_amp[cell_idx, contrast_idx] = compute_4Hz_amp(arr_mean_psth_cycle[cell_idx, :, contrast_idx])

    mdic = {'params': params, 'unique_params': unique_params, 'mean_psth_cycle': arr_mean_psth_cycle,
            '4Hz_amp': arr_4Hz_amp, 'cluster_id': cluster_id, 'pre_pts': np.unique(pre_pts), 
            'stim_pts': np.unique(stim_pts), 'tail_pts': np.unique(tail_pts), 'bin_rate': bin_rate}
    if b_savefull:
        mdic['avg_psth'] = avg_psth
    return mdic

def populate_ct(df_crf, df_ct):
    # Add cell_type column to df_crf and populate it with the cell types from df_ct
    df_crf['cell_type'] = 'unknown'
    for n_id in df_crf['cell_id'].unique():
        # Check if n_id is in df_ct
        str_ct = 'unknown'
        if n_id in df_ct['cell_id'].unique():
            str_ct = df_ct.loc[df_ct['cell_id'] == n_id, 'cell_type'].values[0]

        idx = df_crf['cell_id'] == n_id
        
        df_crf.loc[idx, 'cell_type'] = str_ct
    return df_crf

def fetch_df_crf(str_date, str_chunk, str_algo='kilosort2', typing_file=None):
    ct_query = (djm.CellType() & f'date_id="{str_date}"' & f'chunk_id="{str_chunk}"' & f'algorithm="{str_algo}"')
    if typing_file is not None:
        ct_query = ct_query & f'typing_file="{typing_file}"'
    df_ct = ct_query.fetch(format='frame').reset_index()

    crf_query = (djm.CRF() &  f'date_id="{str_date}"' & f'chunk_id="{str_chunk}"' & f'algorithm="{str_algo}"')

    df_crf = crf_query.fetch(format='frame')
    c_group = df_crf.groupby(level=['cell_id', 'contrast', 'algorithm', 'temporal_frequency'])
    df_crf = c_group.mean().reset_index()
    df_crf = populate_ct(df_crf, df_ct)

    return df_crf

def plot_crf_mosaic(data, ls_types=['OffP', 'OffM', 'OnP', 'OnM'], n_contrast=0.05,
                    f_label=None):
    typing_file = data.str_classification
    str_date = data.str_experiment
    str_chunk = data.str_chunk
    str_algo = data.str_algo
    ncols = len(ls_types)
    df_crf = fetch_df_crf(str_date, str_chunk, str_algo, typing_file)
    df_crf['contrast'] = df_crf['contrast'].astype(float)
    df_plot = df_crf[df_crf['contrast'] == n_contrast]
    cmap = cm.viridis
    norm = Normalize(vmin=df_plot['crf_f1'].min(), vmax=df_plot['crf_f1'].max())

    f, axs = plt.subplots(1, ncols, figsize=(7*ncols, 5))
    f.text(0.1, 1, f'{str_date} {str_chunk} contrast={n_contrast}', fontsize=12, ha='left', va='center')
    if f_label is not None:
        f.text(0.1, 0.95, f_label, fontsize=12, ha='left', va='top', color='red')
    for idx_t, str_type in enumerate(ls_types):
        ax = axs[idx_t]
        df_type = df_plot[df_plot.cell_type==str_type]
        ls_ids = df_type['cell_id'].values
        ls_f1 = df_type['crf_f1'].values
        
        colors = cmap(norm(ls_f1))

        ax,ells =sp.plot_rfs(data, ls_ids, facecolor=colors, ax=ax, ell_color='k', alpha=0.5, b_zoom=True)

        # Add colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        f.colorbar(sm, ax=ax, label='F1')

        # Labels
        ax.set_title(f'{str_type} n={len(ls_ids)}')
    return axs

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='CRF analysis')
    parser.add_argument('experimentName', type=str, help='Name of experiment (eg. 20220531C)')
    parser.add_argument('-f', '--filenames', nargs='+', type=str, help='List of input datafiles')
    parser.add_argument('--savefull', default=False, type=bool, help='Save full PSTH timecourse (default: False)')
    parser.add_argument('-a','--algorithm', default='kilosort2', type=str, help='Sorting algorithm used (yass or kilosort2)')
    parser.add_argument('-s','--search', default='ContrastResponseGrating', type=str, help='Name of stimulus protocol to analyze')
    parser.add_argument('-g','--group', default=None, type=str, help='Search string for EpochGroup (optional)')
    parser.add_argument('-b','--bin_rate', default=1000.0, type=float, help='Bin rate for spikes')
    parser.add_argument('-r','--sample_rate', default=20000.0, type=float, help='Data sample rate')

    args = parser.parse_args()

    # Get the data paths
    SORT_PATH, JSON_PATH, OUTPUT_PATH = cfg.get_data_paths()
    
    # Set filepath for output file
    str_outputfile = args.experimentName + '_' + args.algorithm + '_'
    if isinstance(args.filenames, list):
        for str_file in args.filenames:
            str_outputfile+= str_file + '_'
        str_outputfile+= 'crf.p'
    else:
        str_outputfile+= args.filenames +  '_crf.p'
    
    filepath = os.path.join(OUTPUT_PATH, str_outputfile)
    print('Saving output to: ' + filepath)

    mdic = fetch_data(args.experimentName, str_search=args.search, str_group=args.group, str_algorithm=args.algorithm,
                      bin_rate=args.bin_rate, sample_rate=args.sample_rate, ls_fnames=args.filenames, b_savefull=args.savefull)

    with open(filepath, 'wb') as out_file:
        pickle.dump(mdic, out_file)
    print('Saved to ' + filepath)
