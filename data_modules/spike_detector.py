import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # Import for 3D plotting

def detector(data_matrix, check_detection=False, sample_rate=1e4, refractory_period=1.5e-3, search_window=1.2e-3, cutoff_frequency=500, global_polarity=False, min_peak_amplitude=0):
    

    refractory_period_dp = refractory_period * sample_rate  # datapoints
    search_window_dp = search_window * sample_rate  # datapoints

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

        # get rebounds on either side of each peak
        rebound = get_rebounds(peak_times, current_trace, search_window_dp)

        # cluster spikes
        clustering_data = np.column_stack((peak_amplitudes, rebound['Left'], rebound['Right']))
        start_matrix = np.array([[np.median(peak_amplitudes), np.median(rebound['Left']), np.median(rebound['Right'])],
                                  [np.max(peak_amplitudes), np.max(rebound['Left']), np.max(rebound['Right'])]])
        kmeans = KMeans(n_clusters=2, init=start_matrix, n_init=1, max_iter=10000)

        try:
            kmeans.fit(clustering_data)
            cluster_index = kmeans.labels_
            centroid_amplitudes = kmeans.cluster_centers_
        except ValueError as e:
            if 'Empty cluster' in str(e):
                kmeans = KMeans(n_clusters=2, n_init=10)
                kmeans.fit(clustering_data)
                cluster_index = kmeans.labels_
                centroid_amplitudes = kmeans.cluster_centers_

        spike_cluster_index = np.argmax(centroid_amplitudes[:, 0])  # find cluster with largest peak amplitude
        non_spike_cluster_index = 1 - spike_cluster_index  # non-spike cluster index
        spike_index_logical = (cluster_index == spike_cluster_index)  # spike_ind_log is logical, length of peaks

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
                plot_clustering_data(peak_amplitudes, rebound, cluster_index, spike_cluster_index, non_spike_cluster_index, current_trace, spike_times[tt], refractory_violations[tt], sigF)
            continue

        # check for refractory violations
        refractory_violations[tt] = np.where(np.diff(spike_times[tt]) < refractory_period_dp)[0] + 1
        ref_violations = len(refractory_violations[tt])
        if ref_violations > 0:
            print(f'Trial {tt + 1}: {ref_violations} refractory violations')

        if check_detection:
            plot_clustering_data(peak_amplitudes, rebound, cluster_index, spike_cluster_index, non_spike_cluster_index, current_trace, spike_times[tt], refractory_violations[tt], sigF)

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


def plot_clustering_data(peak_amplitudes, rebound, cluster_index, spike_cluster_index, non_spike_cluster_index, current_trace, spike_times, refractory_violations, sigF):
    """
    Plot clustering data in 3D and the trace with spikes and refractory violations.

    Parameters:
    - peak_amplitudes: Array of peak amplitudes.
    - rebound: Dictionary with 'Left' and 'Right' rebound values.
    - cluster_index: Cluster assignments for each peak.
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
    ax1.scatter(
        peak_amplitudes[cluster_index == spike_cluster_index],
        rebound['Left'][cluster_index == spike_cluster_index],
        rebound['Right'][cluster_index == spike_cluster_index],
        c='r', label='Spikes'
    )
    ax1.scatter(
        peak_amplitudes[cluster_index == non_spike_cluster_index],
        rebound['Left'][cluster_index == non_spike_cluster_index],
        rebound['Right'][cluster_index == non_spike_cluster_index],
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