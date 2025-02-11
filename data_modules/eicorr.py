import os
import numpy as np
import matplotlib.pyplot as plt
import visionloader as vl
import config as cfg
import tqdm
import pickle
from scipy.interpolate import griddata
import spikeoutputs as so
import spikeplots as sp

SORT_PATH, JSON_PATH, OUTPUT_PATH = cfg.get_data_paths()

class MapWithinNoise(object):
    def __init__(self, vcd: vl.VisionCellDataTable):
        self.vcd = vcd
        self.ids = vcd.get_cell_ids()
        for id in self.ids:
            if 'EI' not in self.vcd.main_datatable[id].keys():
                self.ids.remove(id)
        self.ids = np.array(self.ids)
        self.n_ids = len(self.ids)
        self.d_ei = {id: vcd.get_ei_for_cell(id).ei for id in self.ids}

    def compute_ei_corr(self):
        ei_corr = np.zeros((self.n_ids, self.n_ids))
        ei_corr += np.eye(self.n_ids)
        for i in tqdm.trange(self.n_ids):
            id1 = self.ids[i]
            ei1 = self.d_ei[id1].flatten()
            for j in range(i+1, self.n_ids):
                id2 = self.ids[j]
                ei2 = self.d_ei[id2].flatten()
                ei_corr[i, j] = np.corrcoef(ei1, ei2)[0, 1]
        
        # Fill in lower triangle
        for i in range(self.n_ids):
            for j in range(i+1, self.n_ids):
                ei_corr[j, i] = ei_corr[i, j]
        self.ei_corr = ei_corr

class MapAcrossChunk(object):
    def __init__(self, vcd_src: vl.VisionCellDataTable, str_dest_path: str,
                 str_ei_corr_p: str = None):
        self.vcd_src = vcd_src
        self.str_dest_path = str_dest_path

        # Load dest vcd
        self.vcd_dest = vl.load_vision_data(os.path.dirname(str_dest_path),
                                    os.path.basename(str_dest_path)[:-3],
                                    include_ei=True, include_neurons=True)
        
        self.ids_src = self.vcd_src.get_cell_ids()
        self.ids_dest = self.vcd_dest.get_cell_ids()

        # Debug mode: keep only 10 IDs
        # self.ids_src = self.ids_src[:10]
        # self.ids_dest = self.ids_dest[:10]

        # keep only IDs that have EI key in vcd.main_datatable
        for id in self.ids_src:
            if 'EI' not in self.vcd_src.main_datatable[id].keys():
                self.ids_src.remove(id)
        for id in self.ids_dest:
            if 'EI' not in self.vcd_dest.main_datatable[id].keys():
                self.ids_dest.remove(id)

        self.ids_src = np.array(self.ids_src)
        self.ids_dest = np.array(self.ids_dest)

        # Get source EIs
        self.d_ei_src = {} # id: self.vcd_src.get_ei_for_cell(id).ei for id in self.ids_src
        for id in self.ids_src:
            self.d_ei_src[id] = self.vcd_src.get_ei_for_cell(id).ei
        
        # Get destination EIs
        self.d_ei_dest = {}
        for id in self.ids_dest:
            self.d_ei_dest[id] = self.vcd_dest.get_ei_for_cell(id).ei

        # Get src and dest ei maps
        self.ei_maps_src = compute_ei_energy(self.vcd_src, self.ids_src)
        self.ei_maps_dest = compute_ei_energy(self.vcd_dest, self.ids_dest)

        # Load EI correlation if exists
        self.str_ei_corr_p = str_ei_corr_p
        if str_ei_corr_p is not None:
            if os.path.exists(str_ei_corr_p):
                with open(str_ei_corr_p, 'rb') as f:
                    self.ei_corr = pickle.load(f)


    
    def compute_ei_corr(self):
        ei_corr = np.zeros((len(self.ids_src), len(self.ids_dest)))

        for i in tqdm.trange(ei_corr.shape[0]):
            id_src = self.ids_src[i]
            ei_src = self.d_ei_src[id_src].flatten()
            for j in range(ei_corr.shape[1]):
                id_dest = self.ids_dest[j]
                ei_dest = self.d_ei_dest[id_dest].flatten()
                ei_corr[i, j] = np.corrcoef(ei_src, ei_dest)[0, 1]
        self.ei_corr = ei_corr
        if self.str_ei_corr_p is not None:
            with open(self.str_ei_corr_p, 'wb') as f:
                pickle.dump(ei_corr, f)

    def load_ei_corr(self, str_pkl: str):
        with open(str_pkl, 'rb') as f:
            self.ei_corr = pickle.load(f)



def plot_ei_corr_summary(ei: MapAcrossChunk, n_thresh = 0.8):
    f, axs = plt.subplots(ncols=2, figsize=(8, 4))
    ax = axs[0]
    cm=ax.imshow(ei.ei_corr, vmax=1.0)
    plt.colorbar(cm)
    ax.set_title('EI correlation matrix')
    ax.set_xlabel('Destination IDs')
    ax.set_ylabel('Source IDs')

    ax = axs[1]
    # For each column, get the highest value
    max_corr = np.max(ei.ei_corr, axis=0)
    ax.hist(max_corr, bins=len(max_corr)//4)
    ax.axvline(n_thresh, color='r')
    n_matches = np.sum(max_corr > n_thresh)
    n_total = len(max_corr)
    ax.set_xlim(right=1.0)
    ax.set_xlabel('Best match correlation')
    ax.set_title(f'{n_matches}/{n_total} cells above threshold')

def compute_ei_energy(vcd, ids):
    ei_maps = []
    for n_ID in ids:
        ei = vcd.get_ei_for_cell(n_ID).ei
        electrode_map = vcd.get_electrode_map()

        xrange = (np.min(electrode_map[:, 0]), np.max(electrode_map[:, 0]))
        yrange = (np.min(electrode_map[:, 1]), np.max(electrode_map[:, 1]))

        y_dim = 30 # general EI map scaling
        x_dim = int((xrange[1] - xrange[0])/(yrange[1] - yrange[0]) * y_dim) # x dim is proportional to y dim

        x_e = np.linspace(xrange[0], xrange[1], x_dim)
        y_e = np.linspace(yrange[0], yrange[1], y_dim)

        grid_x, grid_y = np.meshgrid(x_e, y_e)
        grid_x = grid_x.T
        grid_y = grid_y.T

        peak_electrode = np.argmax(np.max(np.abs(ei), axis=1))
        peak_frame = np.argmin(ei[peak_electrode, :])

        ei_energy = np.log10(np.mean(np.power(ei, 2), axis=1) + .000000001)
        ei_energy_grid = griddata(electrode_map, ei_energy, (grid_x, grid_y), method='linear', fill_value=np.median(ei_energy))

        ei_maps.append(ei_energy_grid.T)
    return ei_maps

def print_ei_corr_summary(ei: MapAcrossChunk, data: so.SpikeOutputs, 
                          str_type, n_thresh = 0.8):
    ids_src = data.types.d_main_IDs[str_type]
    ids_src = np.intersect1d(ids_src, ei.ids_src)

    n_matches = 0
    n_mult_matches = 0
    for idx_t_id, t_id in enumerate(ids_src):
        idx_src_id = np.where(ei.ids_src == t_id)[0][0]
        idx_matches = np.where(ei.ei_corr[idx_src_id, :] > n_thresh)[0]
        if len(idx_matches) == 1:
            n_matches += 1
        if len(idx_matches) > 1:
            n_mult_matches += 1
    n_no_matches = len(ids_src) - n_matches - n_mult_matches
    print(f'{n_matches} matches, {n_mult_matches} multiple matches, {n_no_matches} no matches of {len(ids_src)} {str_type} cells')

def format_number(n):
    suffixes = ['', 'k', 'M', 'B', 'T']
    suffix_index = 0
    while n >= 1000 and suffix_index < len(suffixes) - 1:
        n /= 1000
        suffix_index += 1
    return f'{n:.1f}{suffixes[suffix_index]}'


def plot_ei_energy(ei_maps, ids, vcd, axs=None):
    if axs is None:
        ncols = len(ei_maps)
        f, axs = plt.subplots(ncols=ncols, figsize=(4*ncols, 4))
    
    ls_n_spikes = []
    ls_peak_idx = []
    for idx, ei_map in enumerate(ei_maps):
        n_ID = ids[idx]
        ax = axs[idx]
        cm=ax.imshow(ei_map, cmap='hot')

        # Get index of peak and plot lines
        peak_idx = np.unravel_index(np.argmax(ei_map), ei_map.shape)
        ax.plot(peak_idx[1], peak_idx[0], 'bo')
        ax.axhline(peak_idx[0], color='b')
        ax.axvline(peak_idx[1], color='b')
        ls_peak_idx.append(peak_idx)

        n_spikes = len(vcd.get_spike_times_for_cell(n_ID))
        ls_n_spikes.append(n_spikes)

    # Compute EI correlation to ei map with most spikes
    idx_max_spikes = np.argmax(ls_n_spikes)
    ei_base = ei_maps[idx_max_spikes]
    for idx, ei_map in enumerate(ei_maps):
        n_ID = ids[idx]
        ax=axs[idx]
        peak_idx = ls_peak_idx[idx]
        n_spikes = ls_n_spikes[idx]
        n_spikes = format_number(n_spikes)
        str_title = f'ID {n_ID}\nPeak: {peak_idx}\n{n_spikes} sps\n'
        if idx != idx_max_spikes:
            r = np.corrcoef(ei_base.flatten(), ei_map.flatten())[0, 1]
            str_title += f'r: {r:.2f}'
        ax.set_title(str_title)
        
def get_match_IDs(ei: MapAcrossChunk, data: so.SpikeOutputs, ls_types: list,
                  n_thresh: float=0.8):
    
    d_match_IDs = {}
    arr_matches = []
    for str_type in ls_types:
        type_ids = data.types.d_main_IDs[str_type]
        type_ids = np.intersect1d(type_ids, ei.ids_src)
        n_matches = 0
        for i, n_ID in enumerate(type_ids):
            idx_src_id = np.where(ei.ids_src == n_ID)[0][0]
            idx_matches = np.where(ei.ei_corr[idx_src_id, :] > n_thresh)[0]
            if len(idx_matches) == 1:
                n_ID_dest = ei.ids_dest[idx_matches[0]]
                d_match_IDs[n_ID] = n_ID_dest
                arr_matches.append([n_ID_dest, str_type])
                n_matches += 1
        print(f'{str_type}: {n_matches}/{len(type_ids)} matches found')
        
    return d_match_IDs, arr_matches

def plot_ei_analysis(eic: MapAcrossChunk, data: so.SpikeOutputs, 
                     idx_t_id, str_type, 
                     n_ei_maps, p_ei_maps, n_thresh = 0.8,
                     str_save=None):
    n_id = data.types.d_main_IDs[str_type][idx_t_id]
    n_id = int(n_id)

    noise_ids = np.array(eic.vcd_src.get_cell_ids())
    idx_n_id = np.where(noise_ids == n_id)[0][0]

    n_ei = n_ei_maps[idx_n_id]
    n_spikes = len(eic.vcd_src.get_spike_times_for_cell(n_id))

    idx_matches = np.where(eic.ei_corr[idx_n_id, :] > n_thresh)[0]
    if len(idx_matches) == 0:
        # Find the next highest correlation
        idx_matches = [np.argsort(eic.ei_corr[idx_n_id, :])[-1]]
    prot_ids = np.array(eic.vcd_dest.get_cell_ids())
    matched_ids = prot_ids[idx_matches]
    ls_p_eis = [p_ei_maps[i] for i in idx_matches]

    ncols = len(ls_p_eis)+1
    f, axs = plt.subplots(ncols=ncols, nrows=2, figsize=(ncols*5, 8))
    ax = axs[0,0]
    cm = ax.imshow(n_ei, cmap='hot')
    # Get index of peak
    peak_idx = np.unravel_index(np.argmax(n_ei), n_ei.shape)
    ax.plot(peak_idx[1], peak_idx[0], 'o', color='blue')
    ax.axhline(peak_idx[0], color='blue')
    ax.axvline(peak_idx[1], color='blue')
    ax.set_title(f'Noise ID {n_id}\nPeak: {peak_idx}\n{n_spikes} sps\n')

    for idx, ei in enumerate(ls_p_eis):
        ax = axs[0,idx+1]
        idx_p_id = idx_matches[idx]
        p_id = prot_ids[idx_p_id]
        p_spikes = len(eic.vcd_dest.get_spike_times_for_cell(p_id))
        cm = ax.imshow(ei, cmap='hot')
        # Get index of peak
        peak_idx = np.unravel_index(np.argmax(ei), ei.shape)
        ax.plot(peak_idx[1], peak_idx[0], 'o', color='blue')
        ax.axhline(peak_idx[0], color='blue')
        ax.axvline(peak_idx[1], color='blue')
        ax.set_title(f'Prot ID {p_id}\nPeak: {peak_idx}\n{p_spikes} sps\nr: {eic.ei_corr[idx_n_id, idx_p_id]:.2f}')

    ax = axs[1,1]

    sorted_idx = np.argsort(eic.ei_corr[idx_n_id, :])[::-1]
    ax.plot(eic.ei_corr[idx_n_id, sorted_idx], '-o')
    ax.set_ylim(-0.5, 1)
    ax.axhline(n_thresh, color='red')
    ax.set_title(f'EI correlations with Noise ID {n_id}')
    ax.set_xlabel('Protocol clusters')

    ax = axs[1,0]
    ax,ells=sp.plot_rfs(data, data.types.d_main_IDs[str_type], ax=ax, sd_mult=0.8, ell_color='k')
    ax.invert_yaxis()
    # Set idx_t_id ell color to red
    ells[idx_t_id].set_facecolor('red')
    ax.set_title(f'{str_type} RFs')

    if str_save is not None:
        plt.savefig(str_save + f'_{str_type}_{n_id}.png', bbox_inches='tight')
        plt.close()

def plot_ei_analysis_all(eic: MapAcrossChunk, data: so.SpikeOutputs, n_thresh: float=0.8,
                         str_savedir: str=None, ls_types: list=None):
    # TODO highlight near threshold misses
    if ls_types is None:
        ls_types = data.ls_RGC_labels

    if str_savedir is None:
        str_savedir = f'./{data.str_experiment}_ei_matches'
    
    if not os.path.exists(str_savedir):
        os.makedirs(str_savedir)

    
    for str_type in ls_types:
        print(f'Processing {str_type}')
        type_ids = data.types.d_main_IDs[str_type]
        for idx_t_id in range(len(type_ids)):
            noise_id = int(type_ids[idx_t_id])
            str_save = os.path.join(str_savedir, f'{str_type}_{noise_id}')
            plot_ei_analysis(eic, data, idx_t_id, str_type, 
                             eic.ei_maps_src, eic.ei_maps_dest,
                              n_thresh=n_thresh, str_save=str_save)