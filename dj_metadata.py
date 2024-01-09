import datajoint as dj
import glob
import os
import numpy as np
import json
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
class EpochBlock(dj.Manual):
    definition = """
    # Epoch block
    -> Protocol
    data_file  : varchar(200)
    ---
    group_label : varchar(200)
    """

@schema
class SortingChunk(dj.Manual):
    definition = """
    # Sorting chunk
    -> EpochBlock
    data_file  : varchar(200)
    ---
    chunk_id : varchar(200)
    """

@schema
class CellTyping(dj.Manual):
    definition = """
    # Typing of cells for a given sorting chunk
    -> SortingChunk
    chunk_id: varchar(200)
    data_file: varchar(200)
    data_files: varchar(200)
    algorithm: varchar(200)
    b_typing_file_exists: bool
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
class MovingChromaticBar(dj.Manual):
    definition = """
    # Parameters for moving chromatic bar protocol
    -> SortingChunk
    data_file_name: varchar(200)
    ---
    bar_size_x: smallint unsigned
    bar_size_y: smallint unsigned
    n_orientations: smallint unsigned
    orientation: blob
    n_repeats: smallint unsigned  
    """

def search_protocol(str_search):
    str_search = str_search.lower()
    ls_protocol_ids = (Protocol() & 'n_blocks > 0').fetch('protocol_id')
    arr_unique_protocols = np.unique(ls_protocol_ids)
    ls_search_protocols = []
    for str_protocol in arr_unique_protocols:
        if str_search in str_protocol.lower():
            ls_search_protocols.append(str_protocol)
    return ls_search_protocols

def chunk_id_protocol(ls_protocol_ids, format='frame'):
    df_chunk = (SortingChunk() & [f'protocol_id = "{str_protocol_id}"' for str_protocol_id in ls_protocol_ids]).fetch(format='frame')
    df_epoch = (EpochBlock() & [f'protocol_id = "{str_protocol_id}"' for str_protocol_id in ls_protocol_ids]).fetch(format='frame')

    # Join
    df_meta = df_chunk.join(df_epoch)
    return df_meta

def get_new_metadata(ls_json):
    ls_dates = [os.path.basename(str_json).split('.')[0] for str_json in ls_json]

    arr_existingdates = Experiment().fetch('date_id')
    arr_newdates = np.setdiff1d(ls_dates, arr_existingdates)

    return arr_newdates

def load_metadata(str_metadata_dir, verbose=False):
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
    ls_protocol_data = []
    ls_block_data = []

    for str_json in ls_new_json:
        d_pdata = {}
        d_bdata = {}
        str_experiment = os.path.basename(str_json).split('.')[0]
        d_pdata['date_id'] = str_experiment
        d_bdata['date_id'] = str_experiment
        # print(str_experiment)
        with open(str_json) as f:
            json_data = json.load(f)
            for i, protocol in enumerate(json_data['protocol']):
                # print(protocol['label'])
                d_pdata['protocol_id']= protocol['label']
                d_bdata['protocol_id']= protocol['label']

                if isinstance(protocol['group'], dict):
                    group = [protocol['group']]
                else:
                    group = protocol['group']
                
                n_groups = len(group)
                d_pdata['n_groups'] = n_groups
                d_pdata['n_blocks'] = 0
                for d_group in group:
                    try:
                        block = d_group['block']
                        if isinstance(d_group['block'], dict):
                            block = [d_group['block']]
                        else:
                            block = d_group['block']
                        
                        n_blocks = len(block)
                        d_pdata['n_blocks'] += n_blocks
                        d_bdata['group_label'] = d_group['label']

                        for d_block in block:
                            str_datafile = d_block['dataFile']
                            d_bdata['data_file'] = str_datafile.split('/')[-2]
                            ls_block_data.append(d_bdata.copy())
                    # Print any exception
                    except Exception as e:
                        if verbose:
                            print('Error in protocol')                        
                            print(str_experiment)
                            print(protocol['label'])
                            print(d_group['label'])
                            print(e)
                ls_protocol_data.append(d_pdata.copy())

    Protocol.insert(ls_protocol_data, skip_duplicates=True)

    # Add epoch block metadata
    for idx in range(len(ls_block_data)):
        try:
            EpochBlock.insert1(ls_block_data[idx], skip_duplicates=True)
        except:
            if verbose:
                print('Error in adding to EpochBlock')
                print(idx)
                print(ls_block_data[idx])
    
    print(f'Added {len(ls_new_json)} new experiments')
    
    # Find sorting chunk information
    ls_chunk_data = []
    ls_chunk_txt = glob.glob('/Volumes/data-1/data/sorted/*/*chunk*.txt')
    ls_chunk_txt.sort()
    for str_txt in ls_chunk_txt:
        d_chunk_data = {}
        str_experiment = os.path.basename(os.path.dirname(str_txt))
        if str_experiment not in SortingChunk().fetch('date_id'):
            str_idxchunk = os.path.basename(str_txt).split('.')[0].split('_')[-1]
            with open(str_txt) as f:
                ls_files = f.readlines()
            ls_files = [x.strip('\n') for x in ls_files[0].split(' ')]

            d_chunk_data['date_id'] = str_experiment
            for str_file in ls_files:
                d_chunk_data['data_file'] = str_file
                d_chunk_data['chunk_id'] = str_idxchunk
                try:
                    d_chunk_data['protocol_id'] = (EpochBlock() & f'data_file = "{d_chunk_data["data_file"]}"' & f'date_id="{str_experiment}"').fetch('protocol_id')[0]
                    ls_chunk_data.append(d_chunk_data.copy())
                except Exception as e:
                    if verbose:
                        print('Error in SortingChunk information')
                        print(d_chunk_data)
                        print(e)
    SortingChunk.insert(ls_chunk_data, skip_duplicates=True)

def load_typing(ANALYSIS_PATH, verbose=False):
    import sys
    sys.path.append('/Users/riekelabbackup/Desktop/Vyom/mea_data_analysis/protocol_analysis/')
    import celltype_io as ctio

    ls_RGC_labels = ['OnP', 'OffP', 'OnM', 'OffM', 'SBC']
    ls_typing_keys = ['num_on_p', 'num_off_p', 'num_on_m', 'num_off_m', 'num_sbc']

    str_noise_protocol = 'manookinlab.protocols.FastNoise'
    # Get noise chunks
    noise_chunks = (SortingChunk() & {'protocol_id': str_noise_protocol}).fetch()
    arr_typingfiles = CellTyping().fetch('typing_file')

    # For each unique date_id
    for date_id in np.unique(noise_chunks['date_id']):
        d_insert = {'date_id': date_id, 'protocol_id': str_noise_protocol}

        ls_chunks_for_date = noise_chunks[noise_chunks['date_id'] == date_id]

        # For each unique chunk_id
        for chunk_id in np.unique(ls_chunks_for_date['chunk_id']):
            d_insert['chunk_id'] = chunk_id
            ls_chunks_for_cid = ls_chunks_for_date[ls_chunks_for_date['chunk_id'] == chunk_id]

            # If multiple noise chunks, concatenate data_file names
            ls_data_files = [chunk['data_file'] for chunk in ls_chunks_for_cid]
            str_data_files = '_'.join(ls_data_files)
            d_insert['data_file'] = ls_data_files[0]
            d_insert['data_files'] = str_data_files
            
            # Get typing file
            ls_typing_files = glob.glob(os.path.join(ANALYSIS_PATH, date_id, chunk_id, '*', '*.txt'))

            # If typing file doesn't exist, don't insert
            if len(ls_typing_files) == 0 and verbose:
                print(f'No typing file for {date_id} {chunk_id} {str_data_files}')
            
            # If typing file exists, insert with num_cells
            else:                
                for str_typing_file in ls_typing_files:
                    if str_typing_file not in arr_typingfiles:
                        d_insert['algorithm'] = os.path.basename(os.path.dirname(str_typing_file))
                        d_insert['b_typing_file_exists'] = True
                        d_insert['typing_file'] = str_typing_file

                        try:
                            types = ctio.CellTypes(str_typing_file, ls_RGC_labels=ls_RGC_labels)

                            d_insert['num_cells'] = types.arr_types.shape[0] # TODO get actual num cells from paramsf file?
                            d_insert['num_goodcells'] = types.arr_types.shape[0] # TODO ISI rejection

                            for idx, str_key in enumerate(ls_typing_keys):
                                # Check if key is in types.d_main_IDs
                                if ls_RGC_labels[idx] in types.d_main_IDs.keys():
                                    d_insert[str_key] = len(types.d_main_IDs[ls_RGC_labels[idx]])
                                elif verbose:
                                    print(f'No {ls_RGC_labels[idx]} in {date_id} {chunk_id} {str_data_files} {str_typing_file}')

                            CellTyping.insert1(d_insert, skip_duplicates=True)
                            print(f'Inserted {date_id} {chunk_id} {str_data_files} {str_typing_file}')

                        except Exception as e:
                            print(f'Error in {date_id} {chunk_id} {str_data_files} {str_typing_file}')
                            print(e)

def make_df_celltyping(df_meta, verbose=False):
    """Construct dataframe of cell typing files from metadata.

    Args:
        df_meta (pd.DataFrame): df_meta output from chunk_id_protocol method
    """
    # get all unique chunks for each date_id
    arr_dates = df_meta.index.get_level_values('date_id').unique().values
    d_chunks = {date_id: df_meta.loc[pd.IndexSlice[date_id, :, :, :], 'chunk_id'].unique() 
                for date_id in arr_dates}
    
    # Create dataframe of cell typing for those chunks
    ls_df_ct = []
    arr_dates = list(d_chunks.keys())
    for date_id in arr_dates:
        arr_chunks = d_chunks[date_id]
        for chunk_id in arr_chunks:
            df_ct = (CellTyping() & f"date_id='{date_id}'" & f"chunk_id='{chunk_id}'").fetch(format='frame')
            if df_ct.shape[0] > 0:
                ls_df_ct.append(df_ct)
            elif verbose:
                print(f"no cell typing for {date_id} {chunk_id}")

    df_ct = pd.concat(ls_df_ct)

    return df_ct

