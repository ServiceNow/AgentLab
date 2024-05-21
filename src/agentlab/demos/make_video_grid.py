from collections import defaultdict
import datetime
from pathlib import Path
from agentlab.experiments.exp_utils import RESULTS_DIR

from moviepy.editor import (
    VideoFileClip,
    clips_array,
    CompositeVideoClip,
    concatenate_videoclips,
    vfx,
)
import numpy as np
from scipy.interpolate import interp1d


def clip_video_length(videos: VideoFileClip, max_duration_range=(12, 15)):
    """Cut the end of the video if too long"""
    clips = []
    for video in videos:

        max_duration = np.random.uniform(*max_duration_range)

        if video.duration > max_duration:
            video = video.subclip(0, max_duration)
        clips.append(video)
    return clips


def group_by_categories(videos: list[VideoFileClip], repeat_lists=3):
    """Group videos by categories if it contains the string in the filename."""
    categories = (
        "order",
        "create",
        "sort",
        "filter",
        "dashboard",
        "impersonation",
        "knowledge-base",
        "all-menu",
    )
    groups = defaultdict(list)
    for video in videos:
        for category in categories:
            if category in video.reader.filename.lower():
                groups[category].append(video)

    for category, video_list in groups.items():
        groups[category] = sorted(video_list, key=lambda x: x.duration, reverse=True)

    for _ in range(repeat_lists):
        for category, video_list in groups.items():
            new_video_list = [video_copy(v) for v in video_list]
            groups[category].extend(new_video_list)

    return groups


def video_copy(video: VideoFileClip):
    return VideoFileClip(video.reader.filename)


class VideoGrid:

    def __init__(self, videos, grid_shape, min_duration=60) -> None:
        self.groups = group_by_categories(videos)
        self.grid_shape = grid_shape

        n_videos = grid_shape[0] * grid_shape[1]
        self.video_lists = [list() for _ in range(n_videos)]
        self.durations = [0] * n_videos
        self.group_representation = {group: 0 for group in self.groups.keys()}
        self.min_duration = min_duration

    def least_represented_group(self):
        keys, representation = zip(*self.group_representation.items())
        return keys[np.argmin(representation)]

    def shortest_list(self):
        return np.argmin(self.durations)

    def grid_center(self):
        return (np.prod(self.grid_shape) - 1) // 2

    def grid_coord_to_index(self, x, y):
        return x * self.grid_shape[1] + y

    def from_group_to_grid(self, from_group, to_index, pop_index=0):
        video = self.pop(from_group, pop_index=pop_index)
        if video is None:
            return

        try:
            self.group_representation[from_group] += video.duration
        except KeyError:
            pass  # group is empty
        self.video_lists[to_index].append(video)
        self.durations[to_index] += video.duration

    def find_video(self, str_match):
        for group, videos in self.groups.items():
            for i, video in enumerate(videos):
                if str_match in video.reader.filename:
                    return group, i

    def pop(self, group, pop_index=0):
        video = self.groups[group].pop(pop_index)
        if len(self.groups[group]) == 0:
            print(f"Group {group} is depleted, deleting it.")
            del self.groups[group]
            del self.group_representation[group]
        return video

    def summarize_groups(self):
        for group, videos in self.groups.items():
            print(f"{group}: {len(videos)}, total duration {sum(v.duration for v in videos):.2f}")

    def summarize_grid(self):
        print(f"Grid shape: {self.grid_shape}")
        for i, video_list in enumerate(self.video_lists):
            # convert i to grid coordinates
            x, y = divmod(i, self.grid_shape[1])

            print(
                f"cell {x:2d},{y:2d}: {len(video_list)} videos, total duration {self.durations[i]:.2f}s"
            )

            for v in video_list:
                name = Path(v.reader.filename).parent.name
                name, seed, uid = name.split("_")[-3:]
                name = name.split(".")[-1]
                print(f"    {name}_{seed} ({v.duration:.1f}s)")

    def is_done(self):
        long_enough = all(d >= self.min_duration for d in self.durations)
        return len(self.groups) == 0 or long_enough

    def make_grid(self):
        for i, v_list in enumerate(self.video_lists):
            self.video_lists[i] = concatenate_videoclips(v_list)

        # add a margin to each video
        for i, video in enumerate(self.video_lists):
            self.video_lists[i] = video.margin(2, color=(0, 0, 0))

        self.grid = np.array(self.video_lists).reshape(self.grid_shape).tolist()
        return self.grid


def create_video_grid(video_files, grid_shape=(3, 3), min_duration=15, max_clip_length=15):
    videos = [VideoFileClip(str(file)) for file in video_files]
    videos = clip_video_length(videos, max_duration_range=(max_clip_length, max_clip_length + 3))
    video_grid = VideoGrid(videos, grid_shape, min_duration=min_duration)
    # duplicate dashboard group to have more dashboard
    video_grid.groups["dashboard_2"] = [video_copy(v) for v in video_grid.groups["dashboard"]]

    video_grid.summarize_groups()

    # control exactly which video appears first.
    group, pop_index = video_grid.find_video("order-apple-watch_43_a2ed17")
    video_grid.from_group_to_grid(group, video_grid.grid_center(), pop_index=pop_index)

    # make sure we have 2 dashboard videos when zooming out
    if grid_shape == (5, 5):
        video_grid.from_group_to_grid("dashboard", video_grid.grid_coord_to_index(1, 1))
        video_grid.from_group_to_grid("dashboard", video_grid.grid_coord_to_index(1, 3))

    while not video_grid.is_done():

        group = video_grid.least_represented_group()
        index = video_grid.shortest_list()
        video_grid.from_group_to_grid(group, index)

    video_grid.summarize_grid()
    return video_grid.make_grid()


def apply_zoom_effect(video, time, values, final_size):

    def zoom_func(t):
        f = interp1d(time, values, kind="linear")
        return f(t).item()

    clip = video.resize(video.size).resize(zoom_func).set_position(("center", "center"))
    return CompositeVideoClip([clip], size=final_size)


if __name__ == "__main__":
    # exp_root = RESULTS_DIR / "2024-02-24_21-58-36_demo_maker"# first test with 10 tasks
    # exp_root = RESULTS_DIR / "2024-02-26_12-01-41_demo_maker"  # all workarena tasks
    # exp_root = RESULTS_DIR / "2024-02-27_17-45-50_demo_maker"  # all workarena tasks * 4 seeds

    # all workarena tasks * 3 seeds on new L1 for k24
    exp_root = RESULTS_DIR / "2024-04-30_18-38-10_demo_maker"

    grid_shape = (5, 5)
    min_duration = 130  # approximate video duration in seconds
    max_clip_length = 15

    # pairs of (time, zoom_values) across the video. This yields a piecewise
    # linear function for the zoom value for each time step.

    speed_factor = 2
    time = [0, 5, 7, 12, 14, 1000]
    zoom_values = [1, 1, 1 / 3, 1 / 3, 1 / 5, 1 / 5]

    # grid_shape = (3, 3)
    # zoom_values = [1, 1, 1 / 3, 1 / 3, 1 / 3, 1 / 3]
    # min_duration = 20

    time = [t * speed_factor for t in time]
    min_duration *= speed_factor
    max_clip_length *= speed_factor

    video_files = list(exp_root.glob("**/extracted_clip.mp4"))
    print(f"Making video grid {repr(grid_shape)} from {len(video_files)} videos.")
    video_grid = create_video_grid(
        video_files,
        grid_shape=grid_shape,
        min_duration=min_duration,
        max_clip_length=max_clip_length,
    )
    final_size = video_grid[0][0].size
    video = clips_array(video_grid)

    print(f"Making zoom out effect.")

    video = apply_zoom_effect(video, time, zoom_values, final_size)

    video = video.fx(vfx.speedx, speed_factor)

    print(f"writing video")
    # make a date string for the video file name using todays date
    date_str = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    video.write_videofile(str(exp_root / f"zoomed_clip_{grid_shape}_{date_str}.mp4"))
