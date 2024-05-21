## Make Demo

Run `ui_assistant.py`. Make sure `record_video=True` and `demo_mode="default"`. You
can use [merge chat](#merge-chat) to combine videos of the browser and the chat
window. You can als o use [extract video activity](#extract-video-activity) to
only extract subset of the videos where there is motion. Note, if the page has
motion even if the user is not interacting, it will be detected as activity.

## Making a Video grid

### Run experiments
`exp_configs.py` contains the `"demo_maker"` configuration. This activate
`record_video=True` and the flag   `demo_mode="default"`. Run this using the 
`launch_command.py`, by making sure `demo_maker` is the selected configuration.
The flag `demo_mode` also has option `"only_visible_elements"`, which will
only highlight target elements that are unobstructed. This can come in handy when 
making a demo for web navigation when the agent tries to click on elements obstructed
by a drop-down list for example. In this situation, `demo_mode="default"` will 
highlight an element on which it can't click in red, while `demo_mode="only_visible_elements"`
will not highlight it.

### Merge Chat
Optionnally, merge the chat video using the `merge_chat_window.py` script. This will
merge the chat video with the main video. You need to modify the script to
specify `exp_root`, this will recursively find all experiments and convert what
containts videos.

### Extract Video Activity
Most videos are filled with long unexciting pauses. Use `extract_video.py` to
detect temporal changes. It will try to find segments that corresponds to
action. The signal processing is a bit flimsy. The `convert` fonction contains
various parameters that cand be tuned. But they are not so document. The
function `temporal_changes` takes a long time to run, but it is cached, so the
second time you run, it will be faster. Then the plotting tool will give you a
sense of how well the auto-magic extraction worked.

### Video Grid
Use `make_video_grid.py` to create a grid of videos. This will search for
the `"extracted_clip.mp4"` generated in the previous step. 
