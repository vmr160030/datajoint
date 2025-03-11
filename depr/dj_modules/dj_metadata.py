import datajoint as dj
import glob
import os
import numpy as np
import json
import sys
import pandas as pd

dj.config['database.host'] = '127.0.0.1'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'simple'

schema = dj.schema('vyomr_tutorial')
@schema
class Experiment(dj.Manual):
    definition = """
    # Experiment date
    date_id         : varchar(9) # date id of experiment
    ---
    animal_id=''       : varchar(20) # animal id
    # experimenter  : varchar(100) # experimenters
    """

@schema
class Protocol(dj.Manual):
    definition = """
    # Protocol
    -> Experiment
    protocol_id     : varchar(200)
    ---
    n_groups        : int
    n_blocks        : int
    """

@schema
class EpochGroup(dj.Manual):
    definition = """
    # Epoch group is a collection of epoch blocks, which is a collection of individual epochs.
    -> Protocol
    group_idx       : int
    group_label     : varchar(200)
    source_label    : varchar(200)
    NDF           : float
    """

@schema
class EpochBlock(dj.Manual):
    definition = """
    # Epoch block is a collection of individual epochs.
    -> EpochGroup
    data_file  : varchar(200)
    block_idx       : int
    ---
    frame_times : longblob
    n_epochs    : int
    """

@schema
class Epoch(dj.Manual):
    definition = """
    # Epoch
    -> EpochBlock
    epoch_idx       : int
    ---
    bath_temperature : float
    backgrounds=null : json
    parameters=null  : json
    """

@schema
class SortingChunk(dj.Manual):
    definition = """
    # Sorting chunk
    -> Experiment
    chunk_id : varchar(200)
    """

@schema
class DataFile(dj.Manual):
    definition = """
    # Data file
    -> EpochBlock
    -> SortingChunk
    data_file: varchar(200)
    """

# @schema
# class SortingAlgo(dj.Manual):
#     definition = """
#     # Sorting algorithm
#     -> SortingChunk
#     algorithm: varchar(200)
#     ---
#     """

@schema
class CellTyping(dj.Manual):
    definition = """
    # Typing of cells for a given sorting chunk
    -> DataFile.proj(noise_data_file='data_file')
    noise_data_files: varchar(200) # All data files included in WN analysis
    algorithm: varchar(200) # Spike Sorting algorithm
    b_typing_file_exists: int # 0: false, 1: true
    typing_file: varchar(200)
    ---
    num_cells=0 : int
    num_goodcells=0 : int
    num_on_p=0 : int
    num_off_p=0 : int
    num_on_m=0 : int
    num_off_m=0 : int
    num_sbc=0 : int
    """

@schema
class TypingNotes(dj.Manual):
    definition = """
    # Notes on cell typing
    -> CellTyping
    ---
    quality: varchar(200)
    typing_status=null: varchar(200) 
    """

@schema
class Cell(dj.Manual):
    definition = """
    # Cluster
    -> SortingChunk
    algorithm: varchar(200)
    cell_id: int
    """

@schema
class STAFit(dj.Manual):
    definition = """
    # STA fit
    -> SortingChunk
    -> Cell
    ---
    noise_grid_size: float # Size of noise grid in microns
    noise_height: float # Height of display in stixels
    noise_width: float # Width of display in stixels
    x0: float # x center of ellipse fit in stixel space
    y0: float # y center of ellipse fit in stixel space
    sigma_x: float # x standard deviation of ellipse fit in stixel space
    sigma_y: float # y standard deviation of ellipse fit in stixel space
    theta: float # angle of ellipse fit. 
    red_time_course=null: longblob # Red channel time course
    green_time_course=null: longblob # Green channel time course
    blue_time_course=null: longblob # Blue channel time course
    """

@schema
class CellType(dj.Manual):
    definition = """
    # Cell type
    -> CellTyping
    -> Cell
    ---
    cell_type: varchar(200)
    """

@schema
class SpikeCounts(dj.Manual):
    definition = """
    # Spike counts
    -> DataFile
    -> Cell
    ---
    n_spikes: int
    """

@schema
class InterspikeInterval(dj.Manual):
    definition = """
    # Interspike interval
    -> DataFile
    -> Cell
    ---
    isi_edges: longblob
    isi: longblob
    """

@schema
class CRF(dj.Manual):
    definition = """
    # CRF data
    -> DataFile
    -> Cell
    contrast: decimal(3,2)
    temporal_frequency: decimal(3,1) # Hz, typically 4Hz
    ---
    crf_f1: float
    """
    
@schema
class QCThresh(dj.Manual):
    definition = """
    # Quality control thresholds
    -> CellTyping
    set: int
    protocol_data_file: varchar(200)
    crf_data_file: varchar(200)
    ---
    n_ref_ms=1.5 : float # ISI refractory violation period in ms
    isi_thresh=0.1 : float # percent spikes in refractory period
    ei_corr_thresh=0.75 : float # correlation threshold for EI
    noise_spikes_thresh=80.0 : float # Top percentile to keep for noise spikes
    noise_spikes_bytype=1 : tinyint # 0: false, 1: true
    protocol_spikes_thresh=80.0 : float # Top percentile to keep for protocol spikes
    protocol_spikes_bytype=1 : tinyint # 0: false, 1: true
    crf_f1_thresh=50.0 : float # Top percentile to keep for CRF F1
    crf_f1_bytype=1 : tinyint # 0: false, 1: true
    """

@schema
class ISIViolations(dj.Computed):
    definition = """
    # ISI violations
    -> QCThresh
    -> InterspikeInterval
    ---
    isi_violations: float
    """

@schema
class EICorrelation(dj.Manual):
    definition = """
    # EI correlation
    -> DataFile.proj(noise_data_file='data_file', noise_g_idx='group_idx', noise_b_idx='block_idx', noise_protocol_id='protocol_id')
    -> DataFile.proj(protocol_data_file='data_file', protocol_g_idx='group_idx', protocol_b_idx='block_idx')
    -> Cell
    ---
    ei_corr: float
    """


# @schema
# class QCParams(dj.Manual):
#     definition = """
#     # Quality control parameters
#     -> QCThresh
#     cell_id: int
#     ---
#     cell_type: varchar(200)
#     noise_spikes: float
#     protocol_spikes: float
#     crf_f1: float
#     noise_isi_violations: float
#     protocol_isi_violations: float
#     ei_corr: float
#     """

def search_protocol(str_search):
    str_search = str_search.lower()
    ls_protocol_ids = (Protocol() & 'n_blocks > 0').fetch('protocol_id')
    arr_unique_protocols = np.unique(ls_protocol_ids)
    ls_search_protocols = []
    for str_protocol in arr_unique_protocols:
        if str_search in str_protocol.lower():
            ls_search_protocols.append(str_protocol)
    return ls_search_protocols

def get_new_metadata(ls_json):
    ls_dates = [os.path.basename(str_json).split('.')[0] for str_json in ls_json]

    arr_existingdates = Experiment().fetch('date_id')
    arr_newdates = np.setdiff1d(ls_dates, arr_existingdates)

    return arr_newdates

def load_metadata(str_metadata_dir):
    # Get json files with dates not already in Experiment
    ls_json = glob.glob(os.path.join(str_metadata_dir, '*.json'))
    arr_newdates = get_new_metadata(ls_json)
    ls_new_json = [os.path.join(str_metadata_dir, str_date + '.json') for str_date in arr_newdates]
    ls_new_json.sort()

    # Add experiment metadata
    ls_exp_data = []
    for str_json in ls_new_json:
        str_experiment = os.path.basename(str_json).split('.')[0]
        ls_exp_data.append({'date_id': str_experiment, 'animal_id': ''})
    Experiment.insert(ls_exp_data, skip_duplicates=True)

    # Add protocol metadata
    for str_json in ls_new_json:
        ls_protocol_data = []
        ls_group_data = []
        ls_block_data = []
        ls_epoch_data = []
        d_pdata = {}
        d_gdata = {}
        d_bdata = {}
        d_edata = {}
        str_experiment = os.path.basename(str_json).split('.')[0]
        d_pdata['date_id'] = str_experiment
        d_gdata['date_id'] = str_experiment
        d_bdata['date_id'] = str_experiment
        d_edata['date_id'] = str_experiment
        print(str_experiment)
        with open(str_json) as f:
            json_data = json.load(f)
            for i, protocol in enumerate(json_data['protocol']):
                # print(protocol['label'])
                d_pdata['protocol_id']= protocol['label']
                d_bdata['protocol_id']= protocol['label']
                d_gdata['protocol_id']= protocol['label']
                d_edata['protocol_id']= protocol['label']

                if isinstance(protocol['group'], dict):
                    group = [protocol['group']]
                else:
                    group = protocol['group']
                
                n_groups = len(group)
                d_pdata['n_groups'] = n_groups
                d_pdata['n_blocks'] = 0
                for g_idx, d_group in enumerate(group):
                    if 'block' not in d_group.keys():
                        d_gdata['group_idx'] = g_idx
                        d_gdata['group_label'] = d_group['label']
                        d_gdata['source_label'] = ''
                        ls_group_data.append(d_gdata.copy())
                        continue

                    block = d_group['block']
                    if isinstance(d_group['block'], dict):
                        block = [d_group['block']]
                    else:
                        block = d_group['block']
                    
                    n_blocks = len(block)
                    d_pdata['n_blocks'] += n_blocks

                    d_gdata['group_idx'] = g_idx
                    d_bdata['group_idx'] = g_idx
                    d_edata['group_idx'] = g_idx

                    d_gdata['group_label'] = d_group['label']
                    if 'source' in d_group.keys():
                        d_gdata['source_label'] = d_group['source']['label']
                    else:
                        d_gdata['source_label'] = ''

                    for b_idx, d_block in enumerate(block):
                        epoch = d_block['epoch']
                        if isinstance(epoch, dict):
                            epoch = [epoch]
                        else:
                            epoch = epoch
                        
                        d_bdata['block_idx'] = b_idx
                        d_edata['block_idx'] = b_idx
                        try:
                            d_bdata['frame_times'] = np.array(d_block['frameTimesMs'])
                        except:
                            d_bdata['frame_times'] = np.array([])
                        d_bdata['n_epochs'] = len(epoch)

                        
                        if len(d_block['epoch']) == 0:
                            continue
                        str_datafile = d_block['dataFile']
                        d_bdata['data_file'] = str_datafile.split('/')[-2]
                        d_edata['data_file'] = str_datafile.split('/')[-2]

                        for e_idx, d_epoch in enumerate(epoch):
                            d_edata['epoch_idx'] = e_idx
                            d_edata['bath_temperature'] = 0#d_epoch['properties']['bathTemperature']
                            # d_edata['backgrounds'] = d_epoch['backgrounds']
                            d_edata['parameters'] = d_epoch['parameters']
                            # d_gdata['NDF'] = d_epoch['parameters']['background:FilterWheel:NDF']
                            ls_epoch_data.append(d_edata.copy())
                            
                        ls_block_data.append(d_bdata.copy())
                    
                    ls_group_data.append(d_gdata.copy())

                ls_protocol_data.append(d_pdata.copy())
        Protocol.insert(ls_protocol_data, skip_duplicates=True)
        EpochGroup.insert(ls_group_data, skip_duplicates=True)
        EpochBlock.insert(ls_block_data, skip_duplicates=True)
        Epoch.insert(ls_epoch_data, skip_duplicates=True)

    # Add epoch block metadata
    # for idx in range(len(ls_block_data)):
    #     try:
    #         EpochBlock.insert1(ls_block_data[idx], skip_duplicates=True)
    #     except:
    #         print('Error in adding to EpochBlock')
    #         print(idx)
    #         print(ls_block_data[idx])
    
    print(f'Added {len(ls_new_json)} new experiments')


def meta_from_protocol(ls_protocol_ids: list)->pd.DataFrame:
    """Get metadata dataframe from list of protocol ids.

    Args:
        ls_protocol_ids (list): List of protocol ids

    Returns:
        pd.DataFrame: Metadata dataframe
    """
    df_chunk = (DataFile() & [f'protocol_id = "{str_protocol_id}"' for str_protocol_id in ls_protocol_ids]).fetch(format='frame')
    df_epoch = (EpochBlock() & [f'protocol_id = "{str_protocol_id}"' for str_protocol_id in ls_protocol_ids]).fetch(format='frame')
    df_egroups = (EpochGroup() & [f'protocol_id = "{str_protocol_id}"' for str_protocol_id in ls_protocol_ids]).fetch(format='frame')

    # Join
    df_meta = df_chunk.join(df_epoch)
    df_meta = df_meta.join(df_egroups)
    return df_meta

def meta_from_date(ls_date_ids: list)->pd.DataFrame:
    """Get metadata dataframe for a list of date_ids.

    Args:
        ls_protocol_ids (list): List of protocol ids

    Returns:
        pd.DataFrame: Metadata dataframe
    """
    df_chunk = (DataFile() & [f'date_id = "{str_protocol_id}"' for str_protocol_id in ls_date_ids]).fetch(format='frame')
    df_epoch = (EpochBlock() & [f'date_id = "{str_protocol_id}"' for str_protocol_id in ls_date_ids]).fetch(format='frame')
    df_egroups = (EpochGroup() & [f'date_id = "{str_protocol_id}"' for str_protocol_id in ls_date_ids]).fetch(format='frame')

    # Join
    df_meta = df_chunk.join(df_epoch)
    df_meta = df_meta.join(df_egroups)
    return df_meta


def query_epochs(str_param, str_val, str_compare='=', str_date=None, str_datafile=None, str_protocol=None, b_AND=True):
    # Construct epoch table query
    ls_query = [f"parameters->>'$.{str_param}'{str_compare}{str_val}"]
    if str_date is not None:
        ls_query.append(f'date_id="{str_date}"')
    if str_datafile is not None:
        ls_query.append(f'data_file="{str_datafile}"')
    if str_protocol is not None:
        ls_query.append(f'protocol_id="{str_protocol}"')

    if b_AND:
        ls_query = dj.AndList(ls_query)
    return Epoch & ls_query


def meta_from_epochs(epochs):
    meta = epochs.fetch("date_id", "data_file")
    # Get unique date, data_file pairs
    meta = np.array(meta).astype(str)
    dates, data_files = np.unique(meta, axis=1)
    df_block = (EpochBlock() & [f'data_file="{data_file}"' for data_file in data_files] & [f'date_id="{date}"' for date in dates]).fetch(format='frame')
    df_chunk = (DataFile() & [f'data_file="{data_file}"' for data_file in data_files] & [f'date_id="{date}"' for date in dates]).fetch(format='frame')
   

    chunk_data_files = df_chunk.index.get_level_values('data_file')
    if len(chunk_data_files) != len(data_files):
        not_found = np.setdiff1d(data_files, chunk_data_files)
        print(f'Chunk_id not found for data files: {not_found}')

    df_meta = df_chunk.join(df_block)

    return df_meta

def celltyping_from_meta(df_meta, verbose=False):
    """Construct dataframe of cell typing files from metadata.

    Args:
        df_meta (pd.DataFrame): df_meta output from chunk_id_protocol method
    """
    # get all unique chunks for each date_id
    df_meta = df_meta.reset_index()
    arr_dates = df_meta['date_id'].unique()
    d_chunks = {date_id: df_meta[df_meta['date_id']==date_id]['chunk_id'].unique() 
                for date_id in arr_dates}
    
    # Create dataframe of cell typing for those chunks
    ls_df_ct = []
    arr_dates = list(d_chunks.keys())
    n_no_ct_chunk = 0
    n_no_ct_date = 0
    b_no_ct_date = True
    for date_id in arr_dates:
        arr_chunks = d_chunks[date_id]
        for chunk_id in arr_chunks:
            df_ct = (CellTyping() & f"date_id='{date_id}'" & f"chunk_id='{chunk_id}'").fetch(format='frame')
            if df_ct.shape[0] > 0:
                ls_df_ct.append(df_ct)
                b_no_ct_date = False
            else:
                n_no_ct_chunk += 1
                if verbose:
                    print(f"no cell typing for {date_id} {chunk_id}")
        
        if b_no_ct_date:
            n_no_ct_date += 1
        b_no_ct_date = True

    print(f'No cell typing for {n_no_ct_chunk} chunks')
    print(f'No cell typing for {n_no_ct_date} dates')
    if len(ls_df_ct) == 0:
        print('No cell typing found')
        return None
    df_ct = pd.concat(ls_df_ct)
    df_ct = df_ct.reset_index()
    df_ct['quality'] = ''

    # Add quality label from TypingNotes table
    for idx in df_ct.index:
        typing_file = df_ct.loc[idx, 'typing_file']
        try:
            str_quality = (TypingNotes() & f"typing_file='{typing_file}'").fetch('quality')[0]
        except Exception as e:
            str_quality = 'Empty'
            if verbose:
                print(f"No quality found for {df_ct.loc[idx, 'date_id']} {df_ct.loc[idx, 'chunk_id']}")
        df_ct.loc[idx, 'quality'] = str_quality

    # Attach metadata about missing cell typing
    df_ct.n_no_ct_chunk = n_no_ct_chunk
    df_ct.n_no_ct_date = n_no_ct_date

    return df_ct