from browsergym.experiments.loop import ExpResult, yield_all_exp_results
from moviepy.editor import VideoFileClip, CompositeVideoClip, clips_array
from agentlab.experiments.exp_utils import RESULTS_DIR


def get_combined_video(exp_result: ExpResult, background_color=(128, 128, 128)):
    """Return the combined video of chat and task. Create it if it doesn't exist."""
    combined_video_path = exp_result.combined_video_path
    if combined_video_path.exists():
        return VideoFileClip(str(combined_video_path))
    else:
        try:
            chat_clip = VideoFileClip(str(exp_result.chat_video_path))
            task_clip = VideoFileClip(str(exp_result.task_video_path))
        except FileNotFoundError as e:
            print(e)
            return None

        # chat_clip = chat_clip.margin(2, color=background_color)
        # task_clip = task_clip.margin(2, color=background_color)

        chat_clip = chat_clip.set_position((0, 0))
        task_clip = task_clip.set_position((chat_clip.size[0], 0))

        size = (chat_clip.size[0] + task_clip.size[0], max(chat_clip.size[1], task_clip.size[1]))
        combined_clip = CompositeVideoClip(
            [chat_clip, task_clip], size=size, bg_color=background_color
        )

        combined_clip.write_videofile(str(combined_video_path), codec="libx264", bitrate="8000k")
        return combined_clip


def get_combined_video2(exp_result: ExpResult):
    combined_video_path = exp_result.combined_video_path
    if combined_video_path.exists():
        return VideoFileClip(str(combined_video_path))
    else:
        try:
            chat_clip = VideoFileClip(str(exp_result.chat_video_path))
            task_clip = VideoFileClip(str(exp_result.task_video_path))
        except FileNotFoundError as e:
            print(e)
            return None

        # Assuming both clips are to be placed side by side
        chat_clip = chat_clip.set_position("left")
        task_clip = task_clip.set_position("right")

        # Calculate the size for the composite video
        max_height = max(chat_clip.size[1], task_clip.size[1])
        total_width = chat_clip.size[0] + task_clip.size[0]
        size = (total_width, max_height)

        # Create a composite video clip
        combined_clip = CompositeVideoClip([chat_clip, task_clip], size=size)

        # Write the video file in WEBM format using VP9 codec
        combined_clip.write_videofile(
            str(combined_video_path),
            codec="libvpx-vp9",
            preset="fast",  # Faster preset
            ffmpeg_params=["-crf", "35"],  # Higher CRF for faster encoding
        )
        return combined_clip


if __name__ == "__main__":

    exp_root = RESULTS_DIR / "concur_final"
    for exp_result in yield_all_exp_results(exp_root):
        print(exp_result.exp_dir)
        try:
            get_combined_video(exp_result)
        except Exception as e:
            print(e)
