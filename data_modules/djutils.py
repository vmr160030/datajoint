import os
import numpy as np
import pandas as pd
import sys
import spikeoutputs as so
import spikeplots as sp
import h5py
import spike_detector as spdet
from collections import namedtuple
sys.path.append('/Users/riekelabbackup/Desktop/Vyom/gitrepos/samarjit_datajoint/next-app/api/')
import schema
from helpers.utils import NAS_ANALYSIS_DIR
from IPython.display import display


def add_param_column(df: pd.DataFrame, param: str, col: str='epoch_parameters'):
    # Add a column to a dataframe that contains the value of a parameter in a dictionary
    # stored in a column of the dataframe.
    df[param] = np.nan
    for idx in df.index:
        params = df.loc[idx, col]
        if param in params:
            df.at[idx, param] = df.loc[idx, col][param]
        else:
            print(f'Parameter {param} not found in {col} for index {idx}')
    return df


def mea_exp_summary(exp_name: str):
    exp_id = (schema.Experiment() & f'exp_name="{exp_name}"').fetch('id')[0]
    eg_q = schema.EpochGroup() & f'experiment_id={exp_id}'
    sc_q = schema.SortingChunk() & f'experiment_id={exp_id}'
    sc_q = sc_q.proj('chunk_name', 'experiment_id',chunk_id='id')
    eg_sc_q = eg_q.proj(group_label='label', group_id='id') * sc_q
    # eb_q = eg_q.proj(group_label='label',group_id='id') * schema.EpochBlock.proj(group_id='parent_id', data_dir='data_dir', chunk_id='chunk_id')
    eb_q = eg_sc_q * schema.EpochBlock.proj('chunk_id', 'protocol_id','data_dir', 
                                            'start_time', 'end_time',
                                            group_id='parent_id', block_id='id')  
    p_q = eb_q * schema.Protocol.proj(..., protocol_name='name')

    df = p_q.fetch(format='frame').reset_index()
    df = df.sort_values('start_time').reset_index()
    
    # Add column of minutes_since_start
    df['minutes_since_start'] = (df['end_time'] - df['start_time'].min()).dt.total_seconds() / 60
    df['minutes_since_start'] = df['minutes_since_start'].round(2)
    # Add delta_minutes which gives derivative along rows
    df['duration_minutes'] = (df['end_time'] - df['start_time']).dt.total_seconds() / 60
    df['duration_minutes'] = df['duration_minutes'].round(2)


    # For each block_id, get first epoch id and its NDF parameter
    df['NDF'] = np.nan
    for bid in df['block_id'].values:
        ep_q = schema.Epoch() & f'parent_id={bid}'
        ep_id = ep_q.fetch('id')[0]
        params = (schema.Epoch() & f'id={ep_id}').fetch('parameters')[0]
        if 'NDF' in params.keys():
            df.loc[df['block_id']==bid, 'NDF'] = params['NDF']
        else:
            print(f'NDF parameter not found for block_id {bid}')


    

    # Order columns
    ls_order = ['data_dir', 'group_label', 'NDF','chunk_name', 'protocol_name',
    'duration_minutes', 'minutes_since_start', 'start_time', 'end_time',
       'experiment_id', 'group_id', 'block_id', 'chunk_id', 'protocol_id']
    df = df[ls_order]

    return df

def get_epoch_data_from_exp(exp_name: str, ls_params: list=None):
    """
    Given experiment id, get dataframe of epoch metadata.
    Arguments:
    - exp_name: name of the experiment
    - ls_params: list of parameters to extract from epoch_parameters column
    Returns:
    - df: dataframe of epoch metadata
    - df_summary: summary dataframe of cell type, cell ID, protocol, and number of epochs
    - d_epoch_params: dictionary of epoch parameters for each protocol
    """
    # Filter epochgroup by experiment_id, then join on EpochBlock, then join on Epoch
    exp_id = (schema.Experiment() & f'exp_name="{exp_name}"').fetch('id')[0]
    eg_q = schema.EpochGroup() & f'experiment_id={exp_id}'
    eg_q = eg_q.proj(cell_id='parent_id',group_label='label', group_id='id')
    c_q = eg_q * schema.Cell.proj(cell_id='id', cell_label='cell_label', cell_properties='properties')
    eb_q = c_q * schema.EpochBlock.proj('protocol_id', group_id='parent_id', block_id='id')
    p_q = eb_q * schema.Protocol.proj(protocol_name='name')
    e_q = p_q * schema.Epoch.proj(epoch_parameters='parameters', block_id='parent_id', epoch_id='id')
    r_q = e_q * schema.Response.proj(..., epoch_id='parent_id', response_id='id') 
    s_q = r_q * schema.Stimulus.proj(epoch_id='parent_id', stim_h5path='h5path', stim_device_name='device_name')
    df = s_q.fetch(format='frame').reset_index()

    # Add column for cell type if exists in cell_properties
    df['cell_type'] = ''
    for idx in df.index:
        cell_props = df.loc[idx, 'cell_properties']
        if cell_props:
            if 'type' in cell_props:
                df.at[idx, 'cell_type'] = cell_props['type']

    # Add columns for any ls_params
    if ls_params:
        for param in ls_params:
            df[param] = np.nan
            for idx in df.index:
                params = df.loc[idx, 'epoch_parameters']
                if param in params:
                    df.at[idx, param] = df.loc[idx, 'epoch_parameters'][param]
                else:
                    print(f'Parameter {param} not found in epoch_parameters for epoch_id {df.loc[idx, "epoch_id"]}')

    # Get unique epoch_parameters keys for each unique protocol_id
    d_epoch_params = {}
    for protocol_id in df['protocol_id'].unique():
        df_q = df[df['protocol_id'] == protocol_id]
        epoch_params = df_q['epoch_parameters'].values[0]
        protocol_name = df_q['protocol_name'].values[0]
        if len(epoch_params) > 0:
            d_epoch_params[protocol_name] = np.array(list(epoch_params.keys()))
        else:
            d_epoch_params[protocol_name] = np.array([])

    # Move id columns to the end
    ls_id_cols = ['protocol_id', 'cell_id', 'group_id', 'block_id', 'epoch_id', 'response_id']
    ls_order = [col for col in df.columns if col not in ls_id_cols] + ls_id_cols
    df = df[ls_order]

    # Move cell type column to the front
    ls_order = ['cell_type'] + [col for col in df.columns if col != 'cell_type']
    df = df[ls_order]

    # Print summary of cell type, cell ID, protocol, and number of epochs
    df_summary = df.groupby(['cell_type', 'cell_id', 'protocol_name']).agg({'epoch_id': 'count'}).reset_index()
    df_summary = df_summary.rename(columns={'epoch_id': 'num_epochs'})
    # For protocol name, split by '.' and keep last part
    df_summary['protocol_name'] = df_summary['protocol_name'].apply(lambda x: x.split('.')[-1])
    # display(df_summary)
    
    return df, df_summary, d_epoch_params

def construct_patch_data(df: pd.DataFrame, str_protocol: str, 
                         cell_id: int, ls_params: list, str_h5: str,
                         b_spiking: bool=True, 
                         b_load_stim: bool=False,
                         **detector_kwargs):
    """Given a dataframe of epoch data, a protocol name, cell id, and a list of parameters,
    return a named tuple encapsulating data.
     """
    df_q = df[df['protocol_name'].str.contains(str_protocol)]
    df_q = df_q[df_q['cell_id']==cell_id]
    df_stim = df_q[df_q['device_name']=='Amp1']
    df_stim = df_stim.reset_index(drop=True)
    print(f'Found {len(df_stim)} trials for {str_protocol} and cell {cell_id}')

    # Get frame monitor data
    df_frame = df_q[df_q['device_name']=='Frame Monitor']
    df_frame = df_frame.reset_index(drop=True)

    # Add param columns
    for param in ls_params:
        df_stim = add_param_column(df_stim, param, col='epoch_parameters')
    df_stim = add_param_column(df_stim, 'preTime', col='epoch_parameters')
    df_stim = add_param_column(df_stim, 'stimTime', col='epoch_parameters')
    df_stim = add_param_column(df_stim, 'tailTime', col='epoch_parameters')
    df_stim = add_param_column(df_stim, 'frameRate', col='epoch_parameters')
    
    # Collect h5paths
    amp_h5paths = df_stim['h5path'].values
    frame_h5paths = df_frame['h5path'].values
    if b_load_stim:
        # Load stim h5paths from stimulus device
        stim_h5paths = df_stim['stim_h5path'].values
        stim_data = []
    
    # Collect data
    amp_data = []
    frame_data = []
    with h5py.File(str_h5, 'r') as f:
        for h5path in amp_h5paths:
            trace = f[h5path]['data']['quantity']
            amp_data.append(trace)
        
        for h5path in frame_h5paths:
            trace = f[h5path]['data']['quantity']
            frame_data.append(trace)

        if b_load_stim:
            try:
                for h5path in stim_h5paths:
                    trace = f[h5path]['data']['quantity']
                    stim_data.append(trace)
            except Exception as e:
                print(f'Error loading stim data: {e}')
                b_load_stim = False
                stim_data = None
    
    amp_data = np.array(amp_data)
    frame_data = np.array(frame_data)
    print(f'Shape of data: Amp1: {amp_data.shape}, Frame Monitor: {frame_data.shape}')
    if b_load_stim:
        stim_data = np.array(stim_data)
        print(f'Shape of stim data: {stim_data.shape}')
    else:
        stim_data = None

    sample_rate = df_stim['sample_rate'].unique()
    assert len(sample_rate) == 1, 'Multiple sample rates found in Amp1 data'
    sample_rate = float(sample_rate[0])
    f_sample_rates = df_frame['sample_rate'].unique()
    if len(f_sample_rates) != 1:
        print('Multiple sample rates found in Frame Monitor data:')
        print(f_sample_rates)
    else:
        print(f'Single sample rate found in Frame Monitor data: {f_sample_rates[0]} Hz')
    print(f'Sample rate: Amp1: {sample_rate} Hz')

    if b_spiking:
        print('Detecting spikes...')
        spikes, amps, refs = spdet.detector(amp_data, sample_rate=sample_rate, 
                                            **detector_kwargs)

    print('Detecting frame flips...')
    frame_times = []
    for idx in df_frame.index:
        trace = frame_data[idx]
        mean = np.mean(trace)
        crossings = np.where(np.diff(np.sign(trace - mean)))[0]
        frame_times.append(crossings)
    
    # Compute avg frame rate
    frame_rates = []
    for idx in df_frame.index:
        crossings = frame_times[idx]
        if len(crossings) > 1:
            frame_rate = 1 / np.mean(np.diff(crossings)) * float(df_frame.at[idx, 'sample_rate'])
            frame_rates.append(frame_rate)
        else:
            frame_rates.append(0)
    frame_rates = np.array(frame_rates)
    print(f'Found unique frame rates: {np.unique(frame_rates)}')

    if b_spiking:
        # Compute stim_spikes, which is number of spikes in each trial in the stim window
        stim_spikes = []
        for idx in df_stim.index:
            pre_time = df_stim.loc[idx, 'preTime']
            stim_time = df_stim.loc[idx, 'stimTime']
            tail_time = df_stim.loc[idx, 'tailTime']
            onset_time = pre_time
            offset_time = pre_time + stim_time
            # Convert from ms to samples
            onset_time = int(onset_time * sample_rate / 1000)
            offset_time = int(offset_time * sample_rate / 1000)
            # Get spikes in this time window
            ss = spikes[idx]
            ss = ss[(ss >= onset_time) & (ss <= offset_time)]
            
            stim_spikes.append(len(ss))
        stim_spikes = np.array(stim_spikes)
        print(f'Shape of stim_spikes: {stim_spikes.shape}')

    # Make parameter dictionary
    d_params = {}
    for param in ls_params:
        d_params[param] = df_stim[param].values
    d_params['preTime'] = df_stim['preTime'].values
    d_params['stimTime'] = df_stim['stimTime'].values
    d_params['tailTime'] = df_stim['tailTime'].values
    d_params['stage_frame_rate'] = df_stim['frameRate'].values[0]
    print(f'Found stage frame rate: {d_params["stage_frame_rate"]} Hz')

    # Make unique parameter dictionary
    d_u_params = {}
    for param in ls_params:
        d_u_params[param] = np.unique(df_stim[param].values)
    d_u_params['preTime'] = np.unique(df_stim['preTime'].values)
    d_u_params['stimTime'] = np.unique(df_stim['stimTime'].values)
    d_u_params['tailTime'] = np.unique(df_stim['tailTime'].values)
    d_u_params['stage_frame_rate'] = np.unique(df_stim['frameRate'].values)

    # Construct named tuple
    # output = {}
    ls_fields = ['data', 'frame_data', 'frame_times', 'frame_rates', 
    'sample_rate','params', 'u_params']
    if b_spiking:
        ls_fields += ['spikes', 'spike_amps', 'spike_refs', 'stim_spikes']
    
    output = namedtuple('output', ls_fields)
    output.data = amp_data
    output.frame_data = frame_data
    output.frame_times = frame_times
    output.frame_rates = frame_rates
    output.stim_data = stim_data
    output.sample_rate = sample_rate
    output.params = d_params
    output.u_params = d_u_params

    if b_spiking:
        output.spikes = spikes
        output.spike_amps = amps
        output.spike_refs = refs
        output.stim_spikes = stim_spikes
    
    return output




def search_protocol(str_search: str):
    str_search = str_search.lower()
    protocols = schema.Protocol().fetch('name')
    protocols = np.unique(protocols)
    matches = []
    for p in protocols:
        if str_search in p.lower():
            matches.append(p)
    matches = np.array(matches)
    return matches

def mea_meta_from_protocols(ls_protocol_names):
    # Query protocol table
    p_q = schema.Protocol() & [f'name="{protocol}"' for protocol in ls_protocol_names]
    p_ids = p_q.fetch('protocol_id')
    p_q = p_q.proj('protocol_id', protocol_name='name')

    # Query EpochBlock with these protocol IDs, get associated experiment IDs
    # WARNING: do not use EpochGroup ever for protocol_id queries bc of the "no_group_protocol" situation.
    # Thank you to @DRezeanu for pointing this out.
    eb_q = schema.EpochBlock() & [f'protocol_id={p_id}' for p_id in p_ids]
    ex_ids = eb_q.fetch('experiment_id')
    ex_ids = np.unique(ex_ids)

    # Join Experiment, EpochGroup, Protocol
    ex_q = schema.Experiment() & [f'id={ex_id}' for ex_id in ex_ids]
    ex_q = ex_q.proj('exp_name', 'is_mea', experiment_id='id')
    eg_q = schema.EpochGroup() & [f'experiment_id={ex_id}' for ex_id in ex_ids]
    eg_q = eg_q.proj('experiment_id', group_label='label', group_id='id')
    eg_q = p_q * eg_q
    eg_q = eg_q * ex_q

    # Join with EpochBlock
    eb_q = eg_q * schema.EpochBlock.proj('chunk_id', 'data_dir', 'protocol_id',
                                         group_id='parent_id', block_id='id')
    
    # Join with SortingChunk and fetch
    sc_q = schema.SortingChunk.proj('chunk_name', chunk_id='id')
    eb_q = eb_q * sc_q
    df = eb_q.fetch(format='frame').reset_index()

    # For each block_id, get first epoch id and its NDF parameter
    df['NDF'] = np.nan
    for bid in df['block_id'].values:
        ep_q = schema.Epoch() & f'parent_id={bid}'
        ep_id = ep_q.fetch('id')[0]
        params = (schema.Epoch() & f'id={ep_id}').fetch('parameters')[0]
        if 'NDF' in params.keys():
            df.loc[df['block_id']==bid, 'NDF'] = params['NDF']
        # else:
            # print(f'NDF parameter not found for {df.loc[df["block_id"]==bid]}')

    df['data_xxx'] = df['data_dir'].apply(lambda x: x.split('/')[-1])
    
    # order columns
    ls_order = ['data_dir', 'group_label', 'NDF', 'chunk_name', 
                'protocol_name', 'exp_name', 'data_xxx', 'is_mea',
                'experiment_id', 'protocol_id', 'group_id', 'block_id', 'chunk_id']
    df = df[ls_order]
    
    return df


def typing_summary_from_file(typing_file_id: int, ls_cell_types: list=None):
    if ls_cell_types is None:
        ls_cell_types = ['OffP', 'OffM', 'OnP', 'OnM', 'SBC']
    # Query SortedCellType
    sct_q = schema.SortedCellType() & f'file_id={typing_file_id}'
    df = sct_q.fetch(format='frame').reset_index()

    d_summary = {}
    d_summary['total_clusters'] = len(df)
    for str_type in ls_cell_types:
        # Count the number of clusters that contain the cell type.
        # Adding '/' as vision delimits with '/', to avoid matches like OnM with OnMystery
        d_summary[str_type] = df['cell_type'].str.contains(str_type+'/', case=False).sum()
    return d_summary


def cell_typing_from_chunks(ls_chunk_ids, ls_cell_types=None, b_remove_zeros=True):
    if ls_cell_types is None:
        ls_cell_types = ['OffP', 'OffM', 'OnP', 'OnM', 'SBC']
    # Query CellTypeFile
    ctf_q = schema.CellTypeFile() & [f'chunk_id={chunk_id}' for chunk_id in ls_chunk_ids]
    ctf_q = ctf_q.proj('chunk_id', 'algorithm', typing_file_name='file_name',
                       typing_file_id='id')
    # Get associated chunk_name, exp_name
    sc_q = schema.SortingChunk.proj('chunk_name', 'experiment_id', chunk_id='id')
    ctf_q = ctf_q * sc_q
    ex_q = schema.Experiment.proj('exp_name', experiment_id='id')
    ctf_q = ctf_q * ex_q
    df = ctf_q.fetch(format='frame').reset_index()

    # Add cell type summary columns
    df['total_clusters'] = 0
    for str_type in ls_cell_types:
        df[str_type] = 0

    # For each typing file, get summary of cell types
    ls_remove = []
    for idx in df.index:
        file_id = df.loc[idx, 'typing_file_id']
        summary = typing_summary_from_file(file_id, ls_cell_types)
        # Check if all cell types are zero
        b_all_zeros = all([summary[str_type]==0 for str_type in ls_cell_types])
        if b_all_zeros:
            str_print = df.loc[idx, 'exp_name'] + ', ' + df.loc[idx, 'chunk_name'] + ', ' + df.loc[idx, 'typing_file_name']
            print(f'No cell type matches found for {str_print}')
            ls_remove.append(idx)
        
        for key, val in summary.items():
            df.at[idx, key] = val

    if b_remove_zeros:
        df = df.drop(ls_remove)
        df = df.reset_index(drop=True)


    # Order columns
    ls_order = ['exp_name','chunk_name', 'algorithm', 
                'typing_file_name', 'total_clusters'] + \
                ls_cell_types + \
                ['experiment_id',  'chunk_id', 'typing_file_id']
    df = df[ls_order]
    df.attrs['ls_cell_types'] = ls_cell_types
    return df

def mea_data_from_meta(rows: pd.DataFrame, ls_cell_types=None):
    # Given a row from cell_typing_from_chunks
    # Return a SpikeOutputs object
    if ls_cell_types is None:
        ls_cell_types = ['OffP', 'OffM', 'OnP', 'OnM', 'SBC']
    row = rows.iloc[0]
    d_init = {'str_experiment': row['exp_name'],
        'dataset_name': row['algorithm'],
        'str_algo': row['algorithm'],
        'ls_RGC_labels': ls_cell_types}
    d_init['paramsfile'] = os.path.join(NAS_ANALYSIS_DIR, row['exp_name'],
                                        row['chunk_name'], row['algorithm'],
                                        f"{row['algorithm']}.params")
    d_init['str_classification'] = os.path.join(NAS_ANALYSIS_DIR, row['exp_name'],
                                        row['chunk_name'], row['algorithm'],
                                        row['typing_file_name'])
    d_init['ls_filenames'] = rows['data_xxx'].values
    data = so.SpikeOutputs(**d_init)
    return data


def mosaics_from_typing(df_ct: pd.DataFrame, df_meta: pd.DataFrame,
                        analysis_dir: str=None, ls_cell_types: list=None):
    # Input df_ct is output of cell_typing_from_chunks.
    # df_meta is output of mea_meta_from_protocols

    if analysis_dir is None:
        analysis_dir = NAS_ANALYSIS_DIR
    
    if ls_cell_types is None:
        ls_cell_types = df_ct.attrs['ls_cell_types']
        print(ls_cell_types)
    
    df = pd.merge(df_meta, df_ct, on=np.intersect1d(df_meta.columns, df_ct.columns).tolist(), how='inner')
    for tfid in df['typing_file_id'].values:
        rows = df[df['typing_file_id']==tfid]
        data = mea_data_from_meta(rows)
        str_annot = f"{rows['exp_name'].values[0]}, {rows['chunk_name'].values[0]}, {rows['typing_file_name'].values[0]}"
        # Check that cell type IDs were found
        if data.types.no_matches:
            print(f'No cell type matches found for {str_annot}')
            continue

        data.load_sta_from_params()

        # Plot mosaics
        # rf_axs, tc_axs = sp.plot_type_rfs_and_tcs(data)
        rf_axs = sp.plot_type_rfs(data)
        # Add exp and chunk annotation
        
        rf_axs[0].text(0, 1.1, str_annot,
                       transform=rf_axs[0].transAxes, fontsize=12)
        # tc_axs[0].text(0, 1.1, str_annot,
        #                   transform=tc_axs[0].transAxes, fontsize=12)

    return df