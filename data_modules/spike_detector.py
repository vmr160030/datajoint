import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

def detector(data_matrix, check_detection=False, sample_rate=1e4, refractory_period=1.5e-3, search_window=1.2e-3, 
             cutoff_frequency=500, global_polarity=False, min_peak_amplitude=0,
             n_clusters=2, max_trial_length_s=1):
    

    refractory_period_dp = refractory_period * sample_rate  # datapoints
    search_window_dp = search_window * sample_rate  # datapoints
    max_trial_length_dp = int(max_trial_length_s * sample_rate)  # Convert seconds to datapoints
    print(f'Max trial length in data points: {max_trial_length_dp} = {max_trial_length_s} s')

    data_matrix = high_pass_filter(data_matrix, cutoff_frequency, 1/sample_rate)

    n_traces = data_matrix.shape[0]
    spike_times = [[] for _ in range(n_traces)]
    spike_amplitudes = [[] for _ in range(n_traces)]
    refractory_violations = [[] for _ in range(n_traces)]

    for tt in range(n_traces):
        current_trace = data_matrix[tt, :]
        if global_polarity:
            if abs(np.max(data_matrix)) > abs(np.min(data_matrix)):  # flip it over, big peaks down
                current_trace = -current_trace
        else:
            if abs(np.max(current_trace)) > abs(np.min(current_trace)):  # flip it over, big peaks down
                current_trace = -current_trace

        # get peaks
        peak_amplitudes, peak_times = get_peaks(current_trace, -1)  # -1 for negative peaks
        peak_times = peak_times[peak_amplitudes < 0]  # only negative deflections
        peak_amplitudes = np.abs(peak_amplitudes[peak_amplitudes < 0])  # only negative deflections
        print(f'Trial {tt + 1}: Found {len(peak_amplitudes)} peaks')

        # get rebounds on either side of each peak
        rebound = get_rebounds(peak_times, current_trace, search_window_dp)

        # cluster spikes
        clustering_data = np.column_stack((peak_amplitudes, rebound['Left'], rebound['Right']))
        
        if len(current_trace) > max_trial_length_dp:
            num_sections = int(np.ceil(len(current_trace) / max_trial_length_dp))
            section_indices = np.array_split(np.arange(len(current_trace)), num_sections)
            print(f"Trial {tt + 1}: Splitting data into {num_sections} sections for KMeans clustering.")

            cluster_index = np.zeros(len(peak_amplitudes), dtype=int)
            
            # Fix non-spike cluster index to k-1, and spike cluster indices to k-2
            spike_cluster_indices = np.arange(n_clusters - 1)
            non_spike_cluster_index = n_clusters - 1
            spike_index_logical = np.zeros(len(peak_amplitudes), dtype=bool)
            for section in section_indices:
                section_mask = np.isin(peak_times, section)  # Select peaks within the current section
                section_data = clustering_data[section_mask]

                if len(section_data) == 0:
                    continue

                # Initialize KMeans for the section
                start_matrix = np.zeros((n_clusters, 3))
                sorted_peak_indices = np.argsort(section_data[:, 0])[::-1]
                for i in range(n_clusters):
                    if i == 0:
                        start_matrix[i, :] = [np.median(section_data[:, 0]), np.median(section_data[:, 1]), np.median(section_data[:, 2])]
                    else:
                        start_matrix[i, :] = section_data[sorted_peak_indices[i - 1]]

                kmeans = KMeans(n_clusters=n_clusters, init=start_matrix, n_init=1, max_iter=10000)
                try:
                    kmeans.fit(section_data)
                    centroid_amplitudes = kmeans.cluster_centers_
                    # Sort clusters by peak amplitude
                    sorted_indices = np.argsort(centroid_amplitudes[:, 0])[::-1]
                    # For cluster_index, map kmeans.labels_ to values in sorted_indices.
                    # So label=1 with largest peak amplitude will be 0, and so on.
                    labels = np.zeros_like(kmeans.labels_)
                    for i, label in enumerate(sorted_indices):
                        labels[kmeans.labels_ == label] = i
                    cluster_index[section_mask] = labels  # Assign cluster indices to the section
                    spike_index_logical[section_mask] = np.isin(cluster_index[section_mask], spike_cluster_indices)  # Logical array for spikes
                except Exception as e:
                    print(f"Section KMeans failed with error: {e}")
                    cluster_index[section_mask] = non_spike_cluster_index  # Assign to non-spike cluster
                    continue


            
        else:
            # Standard KMeans clustering for shorter traces
            start_matrix = np.zeros((n_clusters, 3))
            sorted_peak_indices = np.argsort(peak_amplitudes)[::-1]
            for i in range(n_clusters):
                if i == 0:
                    start_matrix[i, :] = [np.median(peak_amplitudes), np.median(rebound['Left']), np.median(rebound['Right'])]
                else:
                    start_matrix[i, :] = [peak_amplitudes[sorted_peak_indices[i - 1]], rebound['Left'][sorted_peak_indices[i - 1]], rebound['Right'][sorted_peak_indices[i - 1]]]

            kmeans = KMeans(n_clusters=n_clusters, init=start_matrix, n_init=1, max_iter=10000)
            try:
                kmeans.fit(clustering_data)
                cluster_index = kmeans.labels_
                centroid_amplitudes = kmeans.cluster_centers_
            except ValueError as e:
                print(f"Trial {tt + 1}: KMeans clustering failed with error: {e}")
                continue
            # spike_cluster_index = np.argmax(centroid_amplitudes[:, 0])  # find cluster with largest peak amplitude
            # non_spike_cluster_index = 1 - spike_cluster_index  # non-spike cluster index
            # spike_index_logical = (cluster_index == spike_cluster_index)  # spike_ind_log is logical, length of peaks

            # Sort clusters by peak amplitude
            # print(centroid_amplitudes.shape)
            sorted_indices = np.argsort(centroid_amplitudes[:, 0])[::-1]  # Descending order
            spike_cluster_indices = sorted_indices[:n_clusters - 1]  # Top (k-1) clusters
            non_spike_cluster_index = sorted_indices[n_clusters - 1:]
            spike_index_logical = np.isin(cluster_index, spike_cluster_indices)  # Logical array for spikes

        if min_peak_amplitude > 0:
            temp_amp = peak_amplitudes[spike_index_logical]
            temp_times = peak_times[spike_index_logical]
            indices = np.where(temp_amp > min_peak_amplitude)[0]
            non_spike_amplitudes = peak_amplitudes[~spike_index_logical]
            spike_times[tt] = temp_times[indices]#.tolist()
            spike_amplitudes[tt] = temp_amp[indices]#.tolist()
        else:
            spike_times[tt] = peak_times[spike_index_logical]#.tolist()
            spike_amplitudes[tt] = peak_amplitudes[spike_index_logical]#.tolist()
            non_spike_amplitudes = peak_amplitudes[~spike_index_logical]

        # check for no spikes trace
        sigF = (np.mean(spike_amplitudes[tt]) - np.mean(non_spike_amplitudes)) / np.std(non_spike_amplitudes)

        if sigF < 5:  # no spikes
            spike_times[tt] = np.array([])
            spike_amplitudes[tt] = np.array([])
            refractory_violations[tt] = np.array([])
            print(f'Trial {tt + 1}: no spikes!')
            if check_detection:
                plot_clustering_data(peak_amplitudes, rebound, cluster_index, spike_cluster_indices, non_spike_cluster_index, current_trace, spike_times[tt], refractory_violations[tt], sigF)
            continue

        # check for refractory violations
        refractory_violations[tt] = np.where(np.diff(spike_times[tt]) < refractory_period_dp)[0] + 1
        ref_violations = len(refractory_violations[tt])
        if ref_violations > 0:
            print(f'Trial {tt + 1}: {ref_violations} refractory violations')

        if check_detection:
            plot_clustering_data(peak_amplitudes, rebound, cluster_index, spike_cluster_indices, non_spike_cluster_index, current_trace, spike_times[tt], refractory_violations[tt], sigF)

    # if len(spike_times) == 1:  # return vector not list if only 1 trial
    #     spike_times = spike_times[0]
    #     spike_amplitudes = spike_amplitudes[0]
    #     refractory_violations = refractory_violations[0]
    

    return spike_times, spike_amplitudes, refractory_violations


# def plot_clustering_data(peak_amplitudes, rebound, cluster_index, spike_cluster_index, non_spike_cluster_index, current_trace, spike_times, refractory_violations, sigF):

#     plt.figure()
#     plt.subplot(1, 2, 1)
#     plt.scatter(peak_amplitudes[cluster_index == spike_cluster_index],
#                 rebound['Left'][cluster_index == spike_cluster_index],
#                 rebound['Right'][cluster_index == spike_cluster_index], c='r')
#     plt.scatter(peak_amplitudes[cluster_index == non_spike_cluster_index],
#                 rebound['Left'][cluster_index == non_spike_cluster_index],
#                 rebound['Right'][cluster_index == non_spike_cluster_index], c='k')
#     plt.xlabel('Peak Amplitude')
#     plt.ylabel('L rebound')
#     plt.zlabel('R rebound')
#     plt.view_init(8, 36)
#     plt.subplot(1, 2, 2)
#     plt.plot(current_trace, 'k')
#     plt.scatter(spike_times, current_trace[spike_times], c='r')
#     plt.scatter(spike_times[refractory_violations], current_trace[spike_times[refractory_violations]], c='g')
#     plt.title(f'SpikeFactor = {sigF}')
#     plt.draw()
#     plt.pause(0.1)
#     plt.clf()


def plot_clustering_data(peak_amplitudes, rebound, cluster_index, spike_cluster_indices, non_spike_cluster_indices, current_trace, spike_times, refractory_violations, sigF):
    """
    Plot clustering data in 3D and the trace with spikes and refractory violations.

    Parameters:
    - peak_amplitudes: Array of peak amplitudes.
    - rebound: Dictionary with 'Left' and 'Right' rebound values.
    - cluster_indices: Array of cluster indices.
    - spike_cluster_index: Index of the spike cluster.
    - non_spike_cluster_index: Index of the non-spike cluster.
    - current_trace: The signal trace being analyzed.
    - spike_times: Indices of detected spikes.
    - refractory_violations: Indices of refractory violations.
    - sigF: Spike factor for the current trace.
    """
    fig = plt.figure(figsize=(12, 6))

    # 3D scatter plot for clustering data
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    # ax1.scatter(
    #     peak_amplitudes[cluster_index == spike_cluster_indices],
    #     rebound['Left'][cluster_index == spike_cluster_indices],
    #     rebound['Right'][cluster_index == spike_cluster_indices],
    #     c='r', label='Spikes'
    # )
    # ax1.scatter(
    #     peak_amplitudes[cluster_index == non_spike_cluster_indices],
    #     rebound['Left'][cluster_index == non_spike_cluster_indices],
    #     rebound['Right'][cluster_index == non_spike_cluster_indices],
    #     c='k', label='Non-Spikes'
    # )
    ls_colors = [f'C{i}' for i in range(len(spike_cluster_indices))]
    for i in range(len(spike_cluster_indices)):
        ax1.scatter(
            peak_amplitudes[cluster_index == spike_cluster_indices[i]],
            rebound['Left'][cluster_index == spike_cluster_indices[i]],
            rebound['Right'][cluster_index == spike_cluster_indices[i]],
            c=ls_colors[i], label=f'Spikes {i+1}'
        )
    ax1.scatter(
        peak_amplitudes[cluster_index == non_spike_cluster_indices],
        rebound['Left'][cluster_index == non_spike_cluster_indices],
        rebound['Right'][cluster_index == non_spike_cluster_indices],
        c='k', label='Non-Spikes'
    )
    ax1.set_xlabel('Peak Amplitude')
    ax1.set_ylabel('L Rebound')
    ax1.set_zlabel('R Rebound')
    ax1.view_init(elev=8, azim=36)  # Match MATLAB's view
    ax1.legend()
    ax1.set_title('Clustering Data')

    # 2D plot for the trace
    ax2 = fig.add_subplot(1, 2, 2)
    ax2.plot(current_trace, 'k', label='Trace')
    if len(spike_times) > 0:
        ax2.scatter(spike_times, current_trace[spike_times], c='r', label='Spikes')
    # Check if refractory violations exist before plotting
    if len(refractory_violations) > 0:
        # Print dtype of refractory_violations
        print(f'Refractory violations dtype: {type(refractory_violations[0])}, shape: {np.array(refractory_violations).shape}')
        # return refractory_violations, spike_times, current_trace
        refractory_violations = np.array(refractory_violations).astype(int)
        spike_times = np.array(spike_times).astype(int)
        # print(refractory_violations)
        # print(spike_times)
        ref_sts = spike_times[refractory_violations]
        ref_sts = np.array(ref_sts).astype(int)
        ax2.scatter(spike_times[refractory_violations], current_trace[ref_sts], c='g', label='Refractory Violations')
    ax2.set_title(f'SpikeFactor = {sigF:.2f}')
    ax2.legend()

    plt.tight_layout()
    plt.show()

def get_peaks(X, direction):
    """
    Identify local peaks in the input data based on the specified direction.

    Parameters:
    X : array-like
        Input data (1D array).
    direction : int
        Direction for peak detection; 1 for local maxima, -1 for local minima.

    Returns:
    peaks : array-like
        Values of the detected peaks.
    Ind : array-like
        Indices of the detected peaks.
    """

    if direction > 0:  # local max
        Ind = np.where(np.diff(np.sign(np.diff(X))) < 0)[0] + 1
    else:  # local min
        Ind = np.where(np.diff(np.sign(np.diff(X))) > 0)[0] + 1

    peaks = X[Ind]
    return peaks, Ind

def get_rebounds(peaks_ind, trace, search_interval):
    peaks = trace[peaks_ind]
    r = {'Left': np.zeros_like(peaks), 'Right': np.zeros_like(peaks)}

    for i in range(len(peaks)):
        start_point = max(0, peaks_ind[i] - round(search_interval / 2))
        end_point = min(peaks_ind[i] + round(search_interval / 2), len(trace) - 1)
        
        if peaks[i] < 0:  # negative peaks, look for positive rebounds
            r_left,_ = get_peaks(trace[start_point:peaks_ind[i]], 1)
            r_right,_ = get_peaks(trace[peaks_ind[i]:end_point], 1)
        elif peaks[i] > 0:  # positive peaks, look for negative rebounds
            r_left,_ = get_peaks(trace[start_point:peaks_ind[i]], -1)
            r_right,_ = get_peaks(trace[peaks_ind[i]:end_point], -1)

        r['Left'][i] = r_left[0] if r_left.size > 0 else 0
        r['Right'][i] = r_right[0] if r_right.size > 0 else 0

    return r

def high_pass_filter(X, F, SampleInterval):
    L = X.shape[1] if X.ndim > 1 else len(X)
    if L == 1:  # flip if given a column vector
        X = X.T
        L = X.shape[1]

    FreqStepSize = 1 / (SampleInterval * L)
    FreqKeepPts = round(F / FreqStepSize)

    FFTData = np.fft.fft(X, axis=1)
    FFTData[:, :FreqKeepPts] = 0
    FFTData[:, -FreqKeepPts:] = 0

    Xfilt = np.real(np.fft.ifft(FFTData, axis=1))
    return Xfilt