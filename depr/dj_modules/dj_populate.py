import datajoint as dj
import pandas as pd
import glob
import os
import numpy as np
import json
import visionloader as vl
import dj_metadata as djm
import sys
sys.path.append('../data_modules/')
import crf_analysis as crf
import celltype_io as ctio

dj.config['database.host'] = '127.0.0.1'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'simple'

STR_NAS_PATH = '/Volumes/data-1/'
STR_ANALYSIS_PATH = '/Volumes/data-1/analysis/'
STR_SORT_PATH = '/Volumes/data-1/data/sorted/'

def load_chunks_and_files():
    ls_existing = djm.SortingChunk().fetch('date_id')
    ls_metadates = djm.Experiment().fetch('date_id')

    # Find sorting chunk information
    ls_chunk_data = []
    ls_file_data = []

    str_chunk_dir = os.path.join(STR_NAS_PATH, 'data/sorted/*/*chunk*.txt')
    ls_chunk_txt = glob.glob(str_chunk_dir)

    # Remove any with 'jacked' in the name
    ls_chunk_txt = [x for x in ls_chunk_txt if 'jacked' not in x]

    ls_chunk_txt.sort()
    for str_txt in ls_chunk_txt:
        d_chunk_data = {}
        d_file_data = {}
        str_experiment = os.path.basename(os.path.dirname(str_txt))
        if str_experiment not in ls_existing and str_experiment in ls_metadates:
            str_idxchunk = os.path.basename(str_txt).split('.')[0].split('_')[-1]
            with open(str_txt) as f:
                ls_files = f.readlines()
            ls_files = [x.strip('\n') for x in ls_files[0].split(' ')]

            d_chunk_data['date_id'] = str_experiment
            d_chunk_data['chunk_id'] = str_idxchunk
            ls_chunk_data.append(d_chunk_data.copy())
            
            d_file_data['date_id'] = str_experiment
            for str_file in ls_files:
                d_file_data['chunk_id'] = str_idxchunk
                d_file_data['data_file'] = str_file
                # Check if data_file exists in EpochBlock, i.e. was saved in metadata h5/json
                query = (djm.EpochBlock() & f'data_file = "{d_file_data["data_file"]}"' & f'date_id="{str_experiment}"').fetch()
                if len(query) == 0:
                    continue
                
                d_file_data['protocol_id'] = query['protocol_id'][0]
                d_file_data['group_idx'] = query['group_idx'][0]
                d_file_data['block_idx'] = query['block_idx'][0]
                ls_file_data.append(d_file_data.copy())
        elif str_experiment not in ls_metadates:
            print(f'{str_experiment} not in Experiment table.')

    # Add sorting chunk metadata
    djm.SortingChunk.insert(ls_chunk_data, skip_duplicates=True)

    # Add datafile metadata
    djm.DataFile.insert(ls_file_data, skip_duplicates=True)
    print(f'Added {len(ls_file_data)} new data files')

def load_typing(ANALYSIS_PATH=STR_ANALYSIS_PATH, verbose=False, b_reload_typing_files=False):
    if b_reload_typing_files:
        # Delete all existing typing data
        print(f'Deleting {len(djm.CellTyping())} entries in CellTyping table and regenerating...')
        djm.CellTyping().delete(safemode=False)
        print('Deleted all entries in CellTyping table')
    ls_noise_protocols = ['manookinlab.protocols.FastNoise', 'manookinlab.protocols.SpatialNoise']
    ls_protocol_query = [f'protocol_id = "{str_protocol}"' for str_protocol in ls_noise_protocols]
    ls_RGC_labels = ['OnP', 'OffP', 'OnM', 'OffM', 'SBC']
    ls_label_keys = ['on_p', 'off_p', 'on_m', 'off_m', 'sbc']

    # Get noise chunks
    df_noise = (djm.DataFile() & ls_protocol_query).fetch(format='frame')
    existing_typingfiles = djm.CellTyping().fetch('typing_file')

    # For each unique date_id
    for date_id in df_noise.index.get_level_values('date_id').unique():
        df_date = df_noise.loc[date_id]
        protocol_id = df_date.index.get_level_values('protocol_id').unique()[0]
        d_insert = {'date_id': date_id, 'protocol_id': protocol_id}

        ls_chunks_for_date = df_date.index.get_level_values('chunk_id').unique()

        # For each unique chunk_id
        for chunk_id in ls_chunks_for_date:
            d_insert['chunk_id'] = chunk_id
            df_chunk = df_date.xs(chunk_id, level='chunk_id')
            ls_data_files = df_chunk.index.get_level_values('data_file').unique()
            str_data_files = '_'.join(ls_data_files)
            d_insert['noise_data_file'] = ls_data_files[0]
            d_insert['noise_data_files'] = str_data_files
            d_insert['group_idx'] = df_chunk.index.get_level_values('group_idx').unique()[0]
            d_insert['block_idx'] = df_chunk.index.get_level_values('block_idx').unique()[0]
            
            # Get typing file
            ls_typing_files = glob.glob(os.path.join(ANALYSIS_PATH, date_id, chunk_id, '*', '*.txt'))

            # If typing file doesn't exist, don't insert
            if len(ls_typing_files) == 0 and verbose:
                print(f'No typing file for {date_id} {chunk_id} {str_data_files}')
            
            # If typing file exists, insert with num_cells
            else:
                # Count files not in existing_typingfiles
                n_new = len([x not in existing_typingfiles for x in ls_typing_files])
                if n_new > 0 and verbose:
                    print(f'Found {n_new} new typing files for {date_id} {chunk_id}')
                for str_typing_file in ls_typing_files:
                    if str_typing_file not in existing_typingfiles:
                        d_insert['algorithm'] = os.path.basename(os.path.dirname(str_typing_file))
                        d_insert['b_typing_file_exists'] = True
                        d_insert['typing_file'] = str_typing_file
                        types = ctio.CellTypes(str_typing_file, ls_RGC_labels=ls_RGC_labels)
                        

                        d_insert['num_cells'] = types.arr_types.shape[0] # TODO get actual num cells from paramsf file?
                        d_insert['num_goodcells'] = types.arr_types.shape[0] # TODO ISI rejection

                        for str_label, str_key in zip(ls_RGC_labels, ls_label_keys):
                            if str_label in types.d_main_IDs:
                                d_insert[f'num_{str_key}'] = len(types.d_main_IDs[str_label])

                        djm.CellTyping.insert1(d_insert, skip_duplicates=True)
                        if verbose:
                            print(f'Inserted {date_id} {chunk_id} {str_data_files} {str_typing_file}')
                        
                        ls_cellid_data = []
                        ls_ct_data = []
                        d_cellid_insert = {}
                        d_ct_insert = {}
                        for idx_name in djm.SortingChunk.primary_key:
                            d_cellid_insert[idx_name] = d_insert[idx_name]
                        d_cellid_insert['algorithm'] = d_insert['algorithm']
                        for idx_name in djm.CellTyping.primary_key:
                            d_ct_insert[idx_name] = d_insert[idx_name]
                        
                        for n_ID, str_type in types.d_types.items():
                            d_cellid_insert['cell_id'] = n_ID
                            ls_cellid_data.append(d_cellid_insert.copy())
                            
                            d_ct_insert['cell_id'] = n_ID
                            d_ct_insert['cell_type'] = str_type
                            ls_ct_data.append(d_ct_insert.copy())
                        djm.Cell.insert(ls_cellid_data, skip_duplicates=True)
                        djm.CellType.insert(ls_ct_data, skip_duplicates=True)
    # Print current number of entries in CellTyping
    print(f'There are {len(djm.CellTyping())} entries in CellTyping table')
    if len(djm.CellTyping()) == 0:
        print('No entries! Check that you are connected to NAS and your paths are correct.')

def load_typing_notes(str_csv, verbose=False, b_pop_multiple=False):
    df = pd.read_csv(str_csv)

    # Fill in blank experiment values with unique above
    for i in range(1, len(df)):
        if pd.isna(df['Experiment'][i]):
            df.loc[i, 'Experiment'] = df.loc[i-1, 'Experiment']

    df = df[~df['Quality'].isna()]

    df_ct = djm.CellTyping().fetch(format='frame').reset_index()

    ls_insert = []
    for idx in df_ct.index:
        entry = df_ct.iloc[idx]

        d_insert = {}
        for str_key in djm.CellTyping.primary_key:
            d_insert[str_key] = entry[str_key]

        # Look up this date and chunk in df
        search = df[(df['Experiment'] == entry['date_id']) & (df['Chunk'] == entry['chunk_id'])]
        
        if len(search)==1:
            quality = search['Quality'].values[0]
            typing = search['Typing'].values[0]
            d_insert['quality'] = quality
            if pd.isna(typing):
                typing = 'Unknown'
            d_insert['typing_status'] = typing
            ls_insert.append(d_insert.copy())
        elif len(search)>1 and verbose:
            print(f'Multiple matches found for {entry["date_id"]} {entry["chunk_id"]}')
            if b_pop_multiple:
                for idx in search.index:
                    quality = search['Quality'].values[idx]
                    typing = search['Typing'].values[idx]
                    d_insert['quality'] = quality
                    if pd.isna(typing):
                        typing = 'Unknown'
                    d_insert['typing_status'] = typing
                    ls_insert.append(d_insert.copy())
        elif len(search)==0 and verbose:
            print(f'No match found for {entry["date_id"]} {entry["chunk_id"]}')

    djm.TypingNotes.insert(ls_insert, skip_duplicates=True)
    print(f'Inserted {len(ls_insert)} typing notes')


def add_cellids(str_date, str_chunk, str_algo, ls_ids):
    ls_insert = []
    d_insert = {}
    d_insert['date_id'] = str_date
    d_insert['chunk_id'] = str_chunk
    d_insert['algorithm'] = str_algo
    for id in ls_ids:
        d_insert['cell_id'] = id
        ls_insert.append(d_insert.copy())
    djm.Cell.insert(ls_insert, skip_duplicates=True)


def load_spikecounts(str_exp, str_chunk, str_algo = 'kilosort2', str_sort_dir = STR_SORT_PATH):
    df_files = (djm.DataFile() & f'date_id="{str_exp}"' & f'chunk_id="{str_chunk}"').fetch(format='frame').reset_index()

    n_added = 0
    for idx in df_files.index:
        ls_insert = []
        d_insert = {'date_id': str_exp, 'chunk_id': str_chunk, 'algorithm': str_algo}
        str_prot = df_files['protocol_id'].values[idx]
        str_f = df_files['data_file'].values[idx]
        
        d_insert['protocol_id'] = str_prot
        d_insert['data_file'] = str_f
        d_insert['group_idx']  = df_files['group_idx'].values[idx]
        d_insert['block_idx'] = df_files['block_idx'].values[idx]

        ls_existing = (djm.SpikeCounts() & f'date_id="{str_exp}"' & f'algorithm="{str_algo}"' & f'chunk_id="{str_chunk}"').fetch('data_file')
        ls_existing = np.unique(ls_existing)
        if str_f in ls_existing:
            print(f'{str_f} already in SpikeCounts')
            continue
        str_data_dir = os.path.join(str_sort_dir, str_exp, str_f, str_algo)
        vcd = vl.load_vision_data(analysis_path=str_data_dir, dataset_name=str_f, include_neurons=True)
        ls_ids = vcd.get_cell_ids()
        add_cellids(str_exp, str_chunk, str_algo, ls_ids)

        for cell_id in ls_ids:
            d_insert['cell_id'] = cell_id
            d_insert['n_spikes'] = len(vcd.main_datatable[cell_id]['SpikeTimes'])
            ls_insert.append(d_insert.copy())
        djm.SpikeCounts.insert(ls_insert, skip_duplicates=True)
        n_added += 1

    print(f'Added {n_added} files to SpikeCounts')


def load_sta_fits(str_date, str_chunk, str_algo='kilosort2'):
    query = djm.STAFit() & {'date_id': str_date, 'chunk_id': str_chunk, 'algorithm': str_algo}
    if len(query) > 0:
        print(f'Found {len(query)} entries for {str_date}, {str_chunk}, {str_algo}')
        return
    
    str_paramsfile = os.path.join(STR_SORT_PATH, str_date, str_chunk, str_algo, f'{str_algo}.params')
    if not os.path.exists(str_paramsfile):
        str_paramsfile = os.path.join(STR_ANALYSIS_PATH, str_date, str_chunk, str_algo, f'{str_algo}.params')
    if not os.path.exists(str_paramsfile):
        print(f'No params file found for {str_date} {str_chunk} {str_algo}')
        return
    vcd = vl.load_vision_data(os.path.dirname(str_paramsfile), str_algo, include_params=True, include_runtimemovie_params=True)

    d_insert = {'date_id': str_date, 'chunk_id': str_chunk, 'algorithm': str_algo, 
                'noise_grid_size': vcd.runtimemovie_params.micronsPerStixelX,
                'noise_height': vcd.runtimemovie_params.height, 'noise_width': vcd.runtimemovie_params.width}
    d_keymap = {'x0': 'x0', 'y0': 'y0', 'sigma_x': 'SigmaX', 'sigma_y': 'SigmaY', 'theta': 'Theta',
                'red_time_course': 'RedTimeCourse', 'green_time_course': 'GreenTimeCourse', 'blue_time_course': 'BlueTimeCourse'}
    ls_ids = vcd.get_cell_ids()
    add_cellids(str_date, str_chunk, str_algo, ls_ids)
    ls_insert = []
    for n_id in ls_ids:
        d_insert['cell_id'] = n_id
        for str_dj, str_vcd in d_keymap.items():
            d_insert[str_dj] = vcd.main_datatable[n_id][str_vcd]
        ls_insert.append(d_insert.copy())

    djm.STAFit.insert(ls_insert, skip_duplicates=True)
    print(f'Inserted {len(ls_insert)} cell params into STAFit table')


def load_crf(str_date, str_datafile, str_algo='kilosort2'):
    ls_existing = (djm.CRF() & f'date_id="{str_date}"' & f'data_file="{str_datafile}"' & f'algorithm="{str_algo}"')
    if len(ls_existing) > 0:
        print(f'CRF data already exists for {str_date} {str_datafile}')
        return

    mdic = crf.fetch_data(str_date, ls_fnames=[str_datafile])
    str_protocol = 'manookinlab.protocols.ContrastResponseGrating'
    df_chunk = (djm.DataFile() & f'protocol_id="{str_protocol}"' & f'date_id="{str_date}"' &\
                f'data_file="{str_datafile}"').fetch(format='frame')
    
    ls_crfdata = []
    d_insert = {}
    for str_key in djm.DataFile.primary_key:
        d_insert[str_key] = df_chunk.index.get_level_values(str_key)[0]

    d_insert['algorithm'] = str_algo
    d_insert['temporal_frequency'] = mdic['unique_params']['temporalFrequency'][0]
    add_cellids(str_date, d_insert['chunk_id'], str_algo, mdic['cluster_id'])
    for idx, n_ID in enumerate(mdic['cluster_id']):
        d_insert['cell_id'] = n_ID
        for c_idx, contrast in enumerate(mdic['unique_params']['contrast']):
            d_insert['contrast'] = contrast
            d_insert['crf_f1'] = mdic['4Hz_amp'][idx, c_idx]
            ls_crfdata.append(d_insert.copy())  

    djm.CRF.insert(ls_crfdata, skip_duplicates=True)
    print('CRF data inserted for ' + str_date + ' ' + str_datafile)