import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


def get_transition_bins_by_state_time(data,
                                      frame_rate=58.97,
                                      stride=2,
                                      pre_flash_time=0.4,
                                      verbose=False):
    n_trials = data.stim["n_epochs"]
    pre_time = data.stim["unique_params"]["preTime"][0] * 1e-3
    flash_time = data.stim["unique_params"]["flashTime"][0] * 1e-3
    gap_time = data.stim["unique_params"]["gapTime"][0] * 1e-3
    tail_time = data.stim["unique_params"]["tailTime"][0] * 1e-3
    stim_time = data.stim["unique_params"]["stimTime"][0] * 1e-3

    total_time = pre_time + stim_time + tail_time

    total_frames = np.round(frame_rate * total_time).astype(int)
    frame_time = 1 / frame_rate

    # Generate state.time for each frame
    state_time = np.arange(0, total_frames * frame_time, frame_time)

    n_flashes = int(np.floor(stim_time / (flash_time + gap_time)))

    pre_bins = (state_time < pre_time).sum() * stride
    stim_bins = ((state_time >= pre_time) &
                 (state_time < pre_time + stim_time)).sum() * stride
    tail_bins = (state_time >= pre_time + stim_time).sum() * stride

    total_bins = pre_bins + stim_bins + tail_bins

    state_time -= pre_time
    ls_flash_bins = []
    ls_gap_bins = []
    for i in range(n_trials):
        for j in range(n_flashes):
            flash_bins = ((state_time >= j * (flash_time + gap_time))
                          & (state_time <= (j * (flash_time + gap_time) +
                                            flash_time))).sum()
            flash_bins = flash_bins * stride
            gap_bins = ((state_time >= (j *
                                        (flash_time + gap_time) + flash_time))
                        & (state_time < (j + 1) *
                           (flash_time + gap_time))).sum()
            gap_bins = gap_bins * stride
            ls_flash_bins.append(flash_bins)
            ls_gap_bins.append(gap_bins)

    upper_lim = np.max(ls_flash_bins) + np.max(ls_gap_bins)
    # pre_flash_time must be min of pre_time and input
    pre_flash_time = min(pre_flash_time, pre_time)

    pre_flash_bins = np.round(pre_flash_time * frame_rate).astype(int) * stride

    final_bins = upper_lim + pre_flash_bins

    if verbose:
        print(
            f"Total time: {total_time}s, Total frames: {total_frames}, Frame time: {frame_time:.4f}s"
        )
        print(f"Made state time of shape {state_time.shape}")
        print(f"Number of flashes: {n_flashes}")
        print(
            f"Pre bins: {pre_bins}, Stim bins: {stim_bins}, Tail bins: {tail_bins}"
        )
        print(f"Total bins: {total_bins}")
        print(
            f"Unique flash bins: {np.unique(ls_flash_bins, return_counts=True)}"
        )
        print(f"Unique gap bins: {np.unique(ls_gap_bins, return_counts=True)}")
        print(f"Pre flash time: {pre_flash_time:.4f}s")
        print(f"Pre flash bins: {pre_flash_bins}")

    d_out = {
        'n_trials': n_trials,
        'n_flashes': n_flashes,
        'pre_bins': pre_bins,
        'stim_bins': stim_bins,
        'tail_bins': tail_bins,
        'final_bins': final_bins,
        'pre_flash_bins': pre_flash_bins,
        'flash_bins': np.unique(ls_flash_bins)[0],
        'ls_flash_bins': ls_flash_bins,
        'ls_gap_bins': ls_gap_bins
    }
    return d_out


def reshape_psth_by_state_time(data,
                               n_id,
                               d_bins):
    # Get the spikerate for the given neuron
    # Split up the time axis for each trial into individual image flashes.
    # Each trial has: preTime, (flashTime, gapTime)*n_flashes, tailTime
    n_trials = d_bins['n_trials']
    n_flashes = d_bins['n_flashes']
    pre_flash_bins = d_bins['pre_flash_bins']
    ls_flash_bins = d_bins['ls_flash_bins']
    ls_gap_bins = d_bins['ls_gap_bins']
    final_bins = d_bins['final_bins']

    psth = data.spikes["spike_dict"][n_id]
    reshaped = np.zeros((n_trials, n_flashes, final_bins))

    for i in range(n_trials):
        trial = psth[i]
        lower = 0
        upper = pre_flash_bins
        for j in range(n_flashes):
            flash_bins = ls_flash_bins[j]
            gap_bins = ls_gap_bins[j]

            lower = upper - pre_flash_bins
            upper = lower + flash_bins + gap_bins + pre_flash_bins
            if upper > trial.shape[0]:
                upper = trial.shape[0]
            rec_bins = upper - lower
            reshaped[i, j, :rec_bins] = trial[lower:upper]
    reshaped = reshaped.reshape((-1, final_bins))
    return reshaped


def plot_psth(d_bins, psth):
    plt.imshow(psth, aspect="auto", origin="lower")
    plt.colorbar(label="Spike Rate (Hz)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Image Index")

    pre_flash_bins = d_bins["pre_flash_bins"]
    flash_bins = d_bins["ls_flash_bins"][0]
    plt.axvline(pre_flash_bins - 0.5, color="red")
    plt.axvline(pre_flash_bins+flash_bins - 0.5, color="red")
    # plt.text(pre_flash_bins+flash_bins,
    #          -7,
    #          "Image Offset",
    #          color="red",
    #          ha="center",
    #          va="top",
    #          rotation=90)


# def reshape_psth(data, n_id, gen_rate=58.97, stride=2):
#     # Get the spikerate for the given neuron
#     # Split up the time axis for each trial into individual image flashes.
#     # Each trial has: preTime, (flashTime, gapTime)*n_flashes, tailTime
#     pre_time = data.stim["unique_params"]["preTime"][0] * 1e-3
#     flash_time = data.stim["unique_params"]["flashTime"][0] * 1e-3
#     gap_time = data.stim["unique_params"]["gapTime"][0] * 1e-3
#     tail_time = data.stim["unique_params"]["tailTime"][0] * 1e-3
#     stim_time = data.stim["unique_params"]["stimTime"][0] * 1e-3

#     n_flashes = int(np.floor(stim_time / (flash_time + gap_time))) - 1
#     print(f"Number of flashes: {n_flashes}")
#     bin_rate = data.stim["bin_rate"]
#     pre_bins = np.round(pre_time * gen_rate).astype(int) * stride
#     flash_bins = np.round(flash_time * gen_rate).astype(int) * stride
#     gap_bins = np.round(gap_time * gen_rate).astype(int) * stride
#     # gap_bins = 24
#     tail_bins = np.round(tail_time * gen_rate).astype(int) * stride
#     print(
#         f"Pre bins: {pre_bins}, Flash bins: {flash_bins}, Gap bins: {gap_bins}, Tail bins: {tail_bins}"
#     )
#     n_bins = pre_bins + n_flashes * (flash_bins + gap_bins) + tail_bins
#     print(f"Total bins: {n_bins}")

#     psth = data.spikes["spike_dict"][n_id]
#     n_trials, n_bins = psth.shape
#     print(f"PSTH has {n_trials} trials and {n_bins} bins")
#     reshaped = np.zeros((n_trials, n_flashes, flash_bins + gap_bins))
#     for i in range(n_trials):
#         trial = psth[i, pre_bins:-tail_bins]
#         for j in range(n_flashes):
#             reshaped[i, j, :] = trial[j * (flash_bins + gap_bins):(j + 1) *
#                                       (flash_bins + gap_bins)]

#     return reshaped

def load_and_process_img(str_img,screen_size = np.array([1140, 1824]), # rows, cols
                        magnification_factor = 8,
                        ds_factor: int=3, rescale: bool=True,
                        verbose: bool=False):
    img = cv2.imread(str_img, cv2.IMREAD_GRAYSCALE)
    screen_size = screen_size.astype(int)

    img_size = np.array(img.shape)

    scene_size = img_size * magnification_factor
    scene_size = scene_size.astype(int)
    if verbose:
        print(f'Loaded image size: {img_size}')
        print(f'Scene size: {scene_size}')

    # print(f'img_size: {img_size}')
    # print(f'scene_size: {scene_size}')
    # Scale image to scene size with linear interpolation
    img_resized = cv2.resize(img, tuple(scene_size[::-1]), interpolation=cv2.INTER_LINEAR)

    # scene_position = screen_size / 2
    scene_position = scene_size / 2
    if verbose:
        print(f'Image resized size: {img_resized.shape}')
        print(f'Scene position: {scene_position}')
        print(f'Img dtype: {img_resized.dtype}')

    frame = np.zeros(screen_size, dtype=img_resized.dtype)

    # Compute the top-left corner for placing img_resized
    # top_left = (scene_position - np.array(img_resized.shape) / 2).astype(int)

    # # Compute the region of interest (ROI) in the canvas
    # roi_start = np.maximum(top_left, 0)
    # roi_end = np.minimum(top_left + img_resized.shape, screen_size)

    # # Compute the corresponding region in img_resized
    # img_start = np.maximum(-top_left, 0)
    # img_end = img_start + (roi_end - roi_start)
    # # print(f'Roi_start: {roi_start}, Roi_end: {roi_end}')
    # # print(f'Img_start: {img_start}, Img_end: {img_end}')

    # # Place img_resized into the canvas
    # frame[roi_start[0]:roi_end[0], roi_start[1]:roi_end[1]] = img_resized[img_start[0]:img_end[0], img_start[1]:img_end[1]]

    x_vals = np.arange(-screen_size[1] / 2, screen_size[1] / 2)
    y_vals = np.arange(-screen_size[0] / 2, screen_size[0] / 2)
    x_idx = np.round(scene_position[1] + x_vals).astype(int)
    y_idx = np.round(scene_position[0] + y_vals).astype(int)

    x_good = (x_idx >= 0) & (x_idx < scene_size[1])  #& (x_idx < screen_size[1])
    y_good = (y_idx >= 0) & (y_idx < scene_size[0])  #& (y_idx < screen_size[0])

    assign_idx = np.ix_(np.where(y_good)[0], np.where(x_good)[0])
    frame[assign_idx] = img_resized[y_idx[y_good], :][:, x_idx[x_good]]

    # Downsample by ds factor
    if ds_factor > 1:
        ds_shape = np.array(frame.shape) // ds_factor
        if verbose:
            print(f'Downsampling by {ds_factor} to shape {ds_shape}')
        ds_shape = ds_shape.astype(int)
        frame = cv2.resize(frame, tuple(ds_shape[::-1]), interpolation=cv2.INTER_LINEAR)

    if rescale:
        # Convert from (0,255) to (-1, 1)
        frame = (frame.astype(np.float32) / 255.0) * 2 - 1

    return frame


def load_all_imgs(ls_img_paths, str_base_path, ds_factor=3):
    ls_imgs = []
    for i, str_img in enumerate(ls_img_paths):
        str_img = os.path.join(str_base_path, str_img)
        img = load_and_process_img(str_img, ds_factor=ds_factor)
        ls_imgs.append(img)
        if i % 100 == 0:
            print(f"Loaded {i} images")
    return np.array(ls_imgs)
