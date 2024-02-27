import datajoint as dj
import glob
import os
import numpy as np
import json

dj.config['database.host'] = '127.0.0.1'
dj.config['database.user'] = 'root'
dj.config['database.password'] = 'simple'

schema = dj.schema('vyomr_singlecell')
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
    ---
    group_label     : varchar(200)
    source_label    : varchar(200)
    """

@schema
class EpochBlock(dj.Manual):
    definition = """
    # Epoch block is a collection of individual epochs.
    -> EpochGroup
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
        # print(str_experiment)
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
                    try:
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
                        d_gdata['source_label'] = d_group['source']['label']
                        ls_group_data.append(d_gdata.copy())

                        for b_idx, d_block in enumerate(block):
                            epoch = d_block['epoch']
                            if isinstance(epoch, dict):
                                epoch = [epoch]
                            else:
                                epoch = epoch
                            n_epochs = len(epoch)
                            
                            d_bdata['block_idx'] = b_idx
                            d_edata['block_idx'] = b_idx
                            d_bdata['frame_times'] = np.array(d_block['frameTimesMs'])
                            d_bdata['n_epochs'] = len(epoch)

                            for e_idx, d_epoch in enumerate(epoch):
                                d_edata['epoch_idx'] = e_idx
                                d_edata['bath_temperature'] = d_epoch['properties']['bathTemperature']
                                d_edata['backgrounds'] = d_epoch['backgrounds']
                                d_edata['parameters'] = d_epoch['parameters']
                                ls_epoch_data.append(d_edata.copy())
                            ls_block_data.append(d_bdata.copy())
                    # Print any exception
                    except Exception as e:
                        print('Error in protocol')                        
                        print(str_experiment)
                        print(protocol['label'])
                        print(d_group['label'])
                        print(e)
                ls_protocol_data.append(d_pdata.copy())
        
        Protocol.insert(ls_protocol_data, skip_duplicates=True)
        EpochGroup.insert(ls_group_data, skip_duplicates=True)
        EpochBlock.insert(ls_block_data, skip_duplicates=True)
        Epoch.insert(ls_epoch_data, skip_duplicates=True)

    print(f'Added {len(ls_new_json)} new experiments')

    