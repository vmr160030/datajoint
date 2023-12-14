"""
Parsing cell typing .txt files
Vyom Raval
Thursday, Mar 2, 2023
"""

import numpy as np

class CellTypes(object):
    def __init__(self, str_txt, ls_RGC_labels=['OffP', 'OffM', 'OnP', 'OnM']):
        self.str_txt = str_txt
        self.arr_types = np.loadtxt(str_txt, dtype=str)

        # self.ls_RGC_labels = ['OffP', 'OffM', 'OnP', 'OnM']
        self.ls_RGC_labels = ls_RGC_labels

        # Map cell ID to RGC label
        self.d_types = {}
        for c_idx in range(len(self.arr_types)):
            n_ID = int(self.arr_types[c_idx, 0])
            str_type = self.arr_types[c_idx, 1]
            
            # Check if RGC label match in str_type
            for str_RGC in self.ls_RGC_labels:
                if str_RGC in str_type.split('/'):
                    str_type = str_RGC
                
            # If no match, then retain original str_type
            self.d_types[n_ID] = str_type

        # Get list of IDs for main RGC types
        self.d_main_IDs = {}
        for str_RGC in self.ls_RGC_labels:
            arr_type_ids = self.get_ids_of_type(str_RGC)
            # Add to dictionary if there are any IDs
            if len(arr_type_ids) > 0:
                self.d_main_IDs[str_RGC] = arr_type_ids

    def print_summary(self, b_only_main_types=False):
        print('Total number of cells: {}'.format(len(self.d_types)))
        arr_labels, arr_numcells = np.unique(list(self.d_types.values()), return_counts=True)
        
        if b_only_main_types:
            for str_RGC in self.ls_RGC_labels:
                # Check if the label is present in classification file
                if str_RGC in arr_labels:
                    print('Number of {}: {}'.format(str_RGC, arr_numcells[arr_labels == str_RGC][0]))
                else:
                    print('Number of {}: {}'.format(str_RGC, 0))
        else:
            for type_idx in range(len(arr_labels)):
                print('Number of {}: {}'.format(arr_labels[type_idx], arr_numcells[type_idx]))

    def get_ids_of_type(self, str_type):
        ls_cellids = []
        for n_ID in self.d_types.keys():
            if self.d_types[n_ID] == str_type:
                ls_cellids.append(n_ID)
        return np.array(ls_cellids)
    
def map_ids_to_idx(arr_source_IDs, arr_target_IDs, verbose=False):
    """
    Map cell IDs in arr_source_IDs to indices in arr_target_IDs
    """
    arr_idx = np.zeros(len(arr_source_IDs), dtype=int)
    for idx in range(len(arr_source_IDs)):
        try:
            arr_idx[idx] = np.where(arr_target_IDs == arr_source_IDs[idx])[0][0]
        except:
            if verbose:
                print('Cell ID {} not found in data'.format(arr_source_IDs[idx]))
            arr_idx[idx] = -1 # This will have to be accounted for later               
    return arr_idx