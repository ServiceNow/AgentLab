import joblib
import tqdm
from browsergym.experiments.loop import (
    RESULTS_DIR,
    ExpResult,
    yield_all_exp_results,
)
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx, VideoClip
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import percentile_filter

memory = joblib.Memory(RESULTS_DIR / "cache", verbose=0)


# def merge_intervals(intervals):
#     # First, sort the intervals based on the start times
#     intervals.sort(key=lambda x: x[0])

#     # This will hold the merged intervals
#     merged = []

#     for interval in intervals:
#         # If the list of merged intervals is empty or if the current
#         # interval does not overlap with the previous one, simply append it.
#         if not merged or merged[-1][1] < interval[0]:
#             merged.append(interval)
#         else:
#             # Otherwise, there is overlap, so merge the current and previous intervals
#             merged[-1] = (merged[-1][0], max(merged[-1][1], interval[1]))

#     return merged


def find_segments(exp_result, offset=0, inflate=1):
    """Find action segments in the experiment result."""
    segments = []
    t0 = None
    for step_info in exp_result.steps_info:
        if step_info.action_time_stamps is None:
            continue
        t_start, t_stop = step_info.action_time_stamps
        if t_start > t_stop:
            print("Wargning: t_start > t_stop, skiping this one")
            continue
        if t0 is None:
            t0 = t_start
        t_begin = t_start - t0 + offset - inflate
        t_end = t_stop - t0 + offset + inflate
        segments.append((t_begin, t_end))

    return segments


def extract_segments(
    original_clip: VideoFileClip, segments: list[tuple[int, int]], add_last_seconds=0
) -> VideoFileClip:
    """
    Extracts segments from a video.
    """
    extracted_clips = []

    if add_last_seconds > 0:
        segments.append((original_clip.duration - add_last_seconds, original_clip.duration))
    for start_time, end_time in segments:
        print(f"extracting {start_time} to {end_time}")
        segment_clip = original_clip.subclip(start_time, end_time)
        extracted_clips.append(segment_clip)

    # merge back into a single clip
    return concatenate_videoclips(extracted_clips)


@memory.cache
def temporal_changes(video_path, p=4):
    """Extract the the temporal changes in a video using lp norm of absolute differences."""
    clip = VideoFileClip(video_path)

    fps = clip.fps  # Frames per second
    frame_changes = []
    timestamps = []

    # Initialize previous frame as None
    prev_frame = None
    frame_index = 0

    # use tqdm

    frames = tqdm.tqdm(clip.iter_frames(dtype="float32"), total=int(clip.duration * fps))
    for frame in frames:
        if prev_frame is not None:
            frame_diff = np.abs(frame - prev_frame)
            lp_norm = np.linalg.norm(frame_diff.flatten(), ord=p)
            frame_changes.append(lp_norm)

            # Calculate the timestamp for the current frame and append to timestamps list
            timestamp = frame_index / fps
            timestamps.append(timestamp)

        prev_frame = frame
        frame_index += 1

    clip.close()
    return np.array(frame_changes), np.array(timestamps)


def conv_filter(frame_changes, kernel_size=10):
    """Apply a convolution filter to the frame changes, using a kernel of ones."""
    frame_changes = np.array(frame_changes)

    kernel = np.ones(kernel_size)
    kernel /= kernel.sum()
    return np.convolve(frame_changes, kernel, mode="same")


def find_and_join_segments(
    values,
    timestamps,
    allowed_segments,
    percentile=70,
    max_delta=1,
    min_seg_length=1,
    max_seg_count=20,
):
    """Find and join segments of high values in the input array.

    values: array of values
    timestamps: array of timestamps
    allowed_segments: segments will be truncated or rejected to fit within these
        segments
    percentile: percentile of values to find the trhehsold. This threshold will
        yield the start and stop of the segments.
    max_delta: if 2 consecutive values are separated by less than max_delta,
        they are joined
    min_seg_length: minimum length of a segment to be kept
    max_seg_count: maximum number of segments to keep. longest segments are kept.

    Returns a list of (start, stop) times, identifying the segments.
    """
    timestamps = np.array(timestamps)
    threshold = np.percentile(values, percentile)
    above_threshold_indices = np.where(values > threshold)[0]

    if len(above_threshold_indices) == 0:
        return []  # No values above threshold

    above_threshold_times = timestamps[above_threshold_indices]
    segments = [[above_threshold_times[0]]]

    for i, time in enumerate(above_threshold_times[1:], start=1):
        if above_threshold_indices[i] - above_threshold_indices[i - 1] > 1:
            segments.append([time])
        else:
            segments[-1].append(time)

    joined_segments = [segments[0]]
    for current_segment in segments[1:]:
        if current_segment[0] - joined_segments[-1][-1] <= max_delta:
            joined_segments[-1].extend(current_segment)
        else:
            joined_segments.append(current_segment)

    # Truncate and filter segments based on allowed_segments and min_seg_length
    truncated_segments = []
    for segment in joined_segments:
        segment_start, segment_end = segment[0], segment[-1]
        for allowed_start, allowed_end in allowed_segments:
            if allowed_start <= segment_end and allowed_end >= segment_start:
                # Truncate the segment to fit within the allowed range
                start = max(segment_start, allowed_start)
                end = min(segment_end, allowed_end)
                # Check if the truncated segment meets the minimum length requirement
                if end - start >= min_seg_length:
                    truncated_segments.append((start, end))
                break  # Assumes each segment fits in at most one allowed segment

    # keep only the longest segments.
    if len(truncated_segments) > max_seg_count:
        seg_lengths = [end - start for start, end in truncated_segments]
        longest_segments = np.argsort(seg_lengths)[::-1][:max_seg_count]
        truncated_segments = [
            truncated_segments[i] for i in range(len(truncated_segments)) if i in longest_segments
        ]

    return truncated_segments


def plot_changes(timestamps, changes, changes_filtered, action_masks, segments):
    """Plot the temporal changes and the segments."""
    plt.plot(timestamps, changes, label="changes")
    plt.plot(timestamps, changes_filtered, label="changes_filtered")

    for start, stop in action_masks:
        plt.axvspan(start, stop, color="gray", alpha=0.2)

    for start, stop in segments:
        plt.axvspan(start, stop, color="blue", alpha=0.2)

    plt.xlabel("Time (s)")
    plt.ylabel("Lp Norm")
    plt.title("Temporal Changes")
    plt.legend(loc="upper right")


def convert(
    exp_result: ExpResult,
    just_plot=False,
    trust_time_stamps=True,
    use_combined_video=False,
    keep_beginining=False,
    keep_end=False,
    webm_format=True,
):
    """Convert the video in the experiment directory to a shorter version with
    the segments of interest.
    """
    if use_combined_video:
        video_file = exp_result.combined_video_path
    else:
        video_file = exp_result.task_video_path
    action_masks = find_segments(exp_result, offset=0.2, inflate=0.5)

    if trust_time_stamps:
        segments = action_masks[1:]
        plot_args = None
    else:
        frame_changes, timestamps = temporal_changes(str(video_file), p=6)

        changes_convolved = conv_filter(frame_changes)
        changes_pfiltered = percentile_filter(frame_changes, 85, size=21)
        changes_filtered = 0.7 * changes_pfiltered + 0.3 * changes_convolved

        allowed_segments = action_masks[1:]
        # start of first and end of last
        # allowed_segments = [(allowed_segments[0][0], allowed_segments[-1][1])]

        segments = find_and_join_segments(
            changes_filtered,
            timestamps,
            allowed_segments=allowed_segments,
            percentile=80,
            max_delta=0.5,
            min_seg_length=0.5,
        )

        plot_args = dict(timestamps, frame_changes, changes_filtered, action_masks, segments)

    if not just_plot:
        clip = VideoFileClip(str(video_file))

        if keep_beginining:
            segments[0] = (0, segments[0][1])
        if keep_end:
            segments[-1] = (segments[-1][0], clip.duration)

        extracted_clip = extract_segments(clip, segments, add_last_seconds=6)  # type: VideoClip
        accelerated_clip = extracted_clip.fx(vfx.speedx, 1.5)  # type: VideoClip
        if webm_format:
            accelerated_clip.write_videofile(
                str(exp_result.exp_dir / "extracted_clip.webm"),
                codec="libvpx-vp9",
                preset="fast",  # Faster preset
                ffmpeg_params=["-crf", "35"],  # Higher CRF for faster encoding
            )
        else:
            accelerated_clip.write_videofile(
                str(exp_result.exp_dir / "extracted_clip.mp4"), codec="libx264", bitrate="8000k"
            )

    return plot_args


def plot_traces(plot_args_list):
    plot_args_list = plot_args_list[:25]
    n_traces = len(plot_args_list)
    num_rows = min(5, int(n_traces**0.5))
    num_cols = min(5, (n_traces + num_rows - 1) // num_rows)
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(9, 9))
    for plot_args, ax in zip(plot_args_list, axs.flat):
        if plot_args is not None:
            plot_changes(**plot_args)
    plt.show()


if __name__ == "__main__":

    # exp_root = RESULTS_DIR / "2024-02-24_21-58-36_demo_maker"  # first test with 10 tasks
    # exp_root = RESULTS_DIR / "2024-02-26_12-01-41_demo_maker"  # all workarena tasks
    # exp_root = RESULTS_DIR / "2024-02-27_17-45-50_demo_maker"  # all workarena tasks * 4 seeds

    # all workarena tasks * 3 seeds on new L1 for k24
    exp_root = RESULTS_DIR / "2024-04-30_18-38-10_demo_maker"

    # exp_root = RESULTS_DIR / "concur_final"

    exp_results = []
    for exp_result in yield_all_exp_results(exp_root):
        try:
            video_path = exp_result.task_video_path
            exp_results.append(exp_result)
        except FileNotFoundError as e:
            print(e)

    # exp_results = exp_results[:4]
    num_files = len(exp_results)

    plot_args_list = []
    for i, exp_result in enumerate(exp_results):
        exp_dir = exp_result.exp_dir
        print(f"Processing file {i}/{num_files}, {exp_dir}")

        try:
            plot_args_list.append(
                convert(
                    exp_result,
                    just_plot=False,
                    use_combined_video=False,
                    keep_beginining=False,
                    keep_end=False,
                    webm_format=False,
                )
            )
        except Exception as e:
            print(f"Error processing {exp_dir}")
            print(e)

    plot_traces(plot_args_list)
