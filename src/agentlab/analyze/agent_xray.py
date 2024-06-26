from copy import deepcopy
from logging import warning
from math import e
from agentlab.experiments.exp_utils import RESULTS_DIR
from attr import dataclass
import gradio as gr
from browsergym.experiments.loop import ExpResult, StepInfo
from pathlib import Path

from agentlab.analyze import inspect_results
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from io import BytesIO
from PIL import Image


class ClickMapper:
    def __init__(self, ax: plt.Axes, step_times: list[float]):
        self.ax = ax
        self.step_times = step_times

    def to_time(self, x_pix_coord):
        x_time_coord, _ = self.ax.transData.inverted().transform((x_pix_coord, 0))
        return x_time_coord

    def to_step(self, x_pix_coord):
        x_time_coord = self.to_time(x_pix_coord)
        return np.searchsorted(self.step_times, x_time_coord)


@dataclass
class EpisodeId:
    task_name: str = None
    seed: int = None
    agent: str = None


@dataclass
class StepId:
    episode_id: EpisodeId = None
    step: int = None


@dataclass
class Info:
    results_dir: Path = None
    exp_list_dir: Path = None
    result_df: pd.DataFrame = None
    tasks_df: pd.DataFrame = None
    exp_result: ExpResult = None
    click_mapper: ClickMapper = None
    step: int = None
    active_tab: str = "Screenshot"

    def update_exp_result(self, episode_id: EpisodeId):
        if self.result_df is None or episode_id.task_name is None or episode_id.seed is None:
            self.exp_result = None

        # find unique row for task_name and seed
        result_df = self.result_df.reset_index(inplace=False)
        sub_df = result_df[
            (result_df["env_args.task_name"] == episode_id.task_name)
            & (result_df["env_args.task_seed"] == episode_id.seed)
        ]
        if len(sub_df) == 0:
            self.exp_result = None
            raise ValueError(
                f"Could not find task_name: {episode_id.task_name} and seed: {episode_id.seed}"
            )

        if len(sub_df) > 1:
            warning(
                f"Found multiple rows for task_name: {episode_id.task_name} and seed: {episode_id.seed}. Using the first one."
            )

        exp_dir = sub_df.iloc[0]["exp_dir"]
        print(exp_dir)
        self.exp_result = ExpResult(exp_dir)
        self.step = 0


info = Info()


css = """
.my-markdown {
    max-height: 400px;
    overflow-y: auto;
}
.my-code-view {
    max-height: 300px;
    overflow-y: auto;
}
code {
    white-space: pre-wrap;
}
"""

css_code = """
<style>
    .code-container {
        height: 700px;  /* Set the desired height */
        overflow: auto;  /* Enable scrolling */
    }
</style>
"""


def run_gradio(results_dir: Path):
    """
    Run Gradio on the selected experiments saved at savedir_base.

    """
    global info
    info.results_dir = results_dir

    with gr.Blocks(theme=gr.themes.Soft(), css=css) as demo:
        episode_id = gr.State(value=EpisodeId())
        task_name = gr.State(value=None)
        step_id = gr.State(value=None)

        with gr.Accordion("Select your experiment by choosing a (task, seed, agent) triplet."):
            with gr.Row():

                exp_dir_choice = gr.Dropdown(
                    choices=get_directory_contents(results_dir),
                    label="Experiment Directory",
                    show_label=False,
                    scale=10,
                    container=False,
                )
                refresh_button = gr.Button("↺", scale=0, size="sm")

            task_table_gr = gr.DataFrame(
                height=300, label="Task selector", show_label=False, interactive=False
            )

            with gr.Row():
                seed_gr = gr.Dropdown(
                    label="Seed selector", show_label=True, container=False, interactive=True
                )
                agent_gr = gr.Dropdown(
                    choices=["Single Agent"],
                    value="Single Agent",
                    label="Agent selector",
                    show_label=False,
                    container=False,
                    interactive=True,
                )

            refresh_button.click(
                fn=refresh_exp_dir_choices, inputs=exp_dir_choice, outputs=exp_dir_choice
            )

            # plot = gr.Plot(value=generate_profiling, label="Profiling",
            # show_label=False)

        with gr.Row():
            episode_info = gr.Markdown(label="Episode Info", elem_classes="my-markdown")
            action_info = gr.Markdown(label="Action Info", elem_classes="my-markdown")
            state_error = gr.Markdown(label="Next Step Error", elem_classes="my-markdown")

        profiling_gr = gr.Image(
            label="Profiling", show_label=False, interactive=False, show_download_button=False
        )

        gr.HTML(css_code)
        with gr.Tabs() as tabs:

            with gr.Tab("Screenshot") as tab_screenshot:
                som_or_not = gr.Dropdown(
                    choices=["Raw Screenshots", "SOM Screenshots"],
                    label="Screenshot Type",
                    value="Raw Screenshots",
                    show_label=False,
                    container=False,
                    interactive=True,
                    scale=0,
                )
                screenshot = gr.Image(
                    show_label=False, interactive=False, show_download_button=False
                )

            with gr.Tab("Screenshot Pair") as tab_screenshot_pair:
                with gr.Row():
                    screenshot1 = gr.Image(
                        show_label=False, interactive=False, show_download_button=False
                    )
                    screenshot2 = gr.Image(
                        show_label=False, interactive=False, show_download_button=False
                    )

            with gr.Tab("DOM HTML") as tab_html:
                html_code = gr.Code(
                    lines=50,
                    interactive=False,
                    language="html",
                    show_label=False,
                    elem_classes=["code-container"],
                    visible=True,
                )

            # 8. Render the Pruned HTML
            with gr.Tab("Pruned DOM HTML") as tab_pruned_html:
                pruned_html_code = gr.Code(
                    lines=50,
                    interactive=False,
                    language="html",
                    show_label=False,
                    elem_classes=["code-container"],
                    visible=True,
                )

            # 9. Render the Accessibility Tree
            with gr.Tab("AXTree") as tab_axtree:
                axtree_code = gr.Code(
                    lines=50,
                    interactive=False,
                    language=None,
                    show_label=False,
                    elem_classes=["code-container"],
                    visible=True,
                )

            with gr.Tab("Chat Messages") as tab_chat:
                chat_messages = gr.Markdown()

        exp_dir_choice.change(
            fn=new_exp_dir, inputs=exp_dir_choice, outputs=[task_table_gr, task_name]
        )

        task_table_gr.select(fn=on_select_task, inputs=task_table_gr, outputs=[task_name])
        task_name.change(fn=update_seeds, inputs=[task_name], outputs=[seed_gr])
        seed_gr.change(fn=on_select_seed, inputs=[seed_gr, task_name], outputs=[episode_id])

        episode_id.change(fn=new_episode, inputs=[episode_id], outputs=[profiling_gr, step_id])

        profiling_gr.select(select_step, inputs=[episode_id], outputs=step_id)

        step_id.change(fn=update_step_info, outputs=[episode_info, action_info, state_error])

        # Update all tabs on step change, but only actually update the active
        # tab. This helps keeping the UI responsive when selecting a new step.
        step_id.change(
            fn=if_active("Screenshot")(update_screenshot),
            inputs=som_or_not,
            outputs=screenshot,
        )
        step_id.change(
            fn=if_active("Screenshot Pair", 2)(update_screenshot_pair),
            inputs=som_or_not,
            outputs=[screenshot1, screenshot2],
        )
        step_id.change(fn=if_active("DOM HTML")(update_html), outputs=html_code)
        step_id.change(
            fn=if_active("Pruned DOM HTML")(update_pruned_html), outputs=pruned_html_code
        )
        step_id.change(fn=if_active("AXTree")(update_axtree), outputs=axtree_code)
        step_id.change(fn=if_active("Chat Messages")(update_chat_messages), outputs=chat_messages)

        # In order to handel tabs that were not visible when step was changed,
        # we need to update them individually when the tab is selected
        tab_screenshot.select(fn=update_screenshot, inputs=som_or_not, outputs=screenshot)
        tab_screenshot_pair.select(
            fn=update_screenshot_pair, inputs=som_or_not, outputs=[screenshot1, screenshot2]
        )
        tab_html.select(fn=update_html, outputs=html_code)
        tab_pruned_html.select(fn=update_pruned_html, outputs=pruned_html_code)
        tab_axtree.select(fn=update_axtree, outputs=axtree_code)
        tab_chat.select(fn=update_chat_messages, outputs=chat_messages)

        som_or_not.change(fn=update_screenshot, inputs=som_or_not, outputs=screenshot)

        # keep track of active tab
        tabs.select(tab_select)

    demo.queue()
    demo.launch(server_port=7889)


def tab_select(evt: gr.SelectData):
    global info
    info.active_tab = evt.value


def if_active(tab_name, n_out=1):
    def decorator(fn):
        def wrapper(*args, **kwargs):
            global info
            if info.active_tab == tab_name:
                # print("updating: ", fn.__name__)
                return fn(*args, **kwargs)
            else:
                # print("skipping: ", fn.__name__)
                if n_out == 1:
                    return gr.update()
                elif n_out > 1:
                    return (gr.update(),) * n_out

        return wrapper

    return decorator


def update_screenshot(som_or_not: str):
    global info
    return get_screenshot(info, som_or_not=som_or_not)


def get_screenshot(info: Info, step: int = None, som_or_not: str = "Raw Screenshots"):
    if step is None:
        step = info.step
    try:
        is_som = som_or_not == "SOM Screenshots"
        return info.exp_result.get_screenshot(step, som=is_som)
    except FileNotFoundError:
        return None


def update_screenshot_pair(som_or_not: str):
    global info
    s1 = get_screenshot(info, info.step, som_or_not)
    s2 = get_screenshot(info, info.step + 1, som_or_not)
    return s1, s2


def update_html():
    return get_obs(key="dom_txt", default="No DOM HTML")


def update_pruned_html():
    return get_obs(key="pruned_html", default="No Pruned HTML")


def update_axtree():
    return get_obs(key="axtree_txt", default="No AXTree")


def update_chat_messages():
    global info
    agent_info = info.exp_result.steps_info[info.step].agent_info
    chat_messages = agent_info.get("chat_messages", ["No Chat Messages"])
    messages = []
    for i, m in enumerate(chat_messages):
        messages.append(f"""# Message {i}\n```\n{m}\n```\n\n""")
    return "\n".join(messages)


def select_step(episode_id: EpisodeId, evt: gr.SelectData):
    global info
    step = info.click_mapper.to_step(evt.index[0])
    info.step = step
    return StepId(episode_id, step)


def update_step_info():
    global info
    return [
        get_episode_info(info),
        get_action_info(info),
        get_state_error(info),
    ]


def get_obs(key: str, default=None):
    global info
    obs = info.exp_result.steps_info[info.step].obs
    return obs.get(key, default)


def code(txt):
    # return f"""<pre style="white-space: pre-wrap; word-wrap:
    # break-word;">{txt}</pre>"""
    return f"""```\n{txt}\n```"""


def get_episode_info(info: Info):
    env_args = info.exp_result.exp_args.env_args
    steps_info = info.exp_result.steps_info
    step_info = steps_info[info.step]
    goal = step_info.obs["goal"]
    cum_reward = info.exp_result.summary_info["cum_reward"]
    exp_dir = info.exp_result.exp_dir
    exp_dir_str = f"{exp_dir.parent.name}/{exp_dir.name}"

    info = f"""\
### {env_args.task_name}
### Step {info.step} / {len(steps_info)-1} (Reward: {cum_reward:.1f})

**Goal:**

{code(goal)}

**Task info:**

{code(step_info.task_info)}

**exp_dir:**

<small style="line-height: 1; margin: 0; padding: 0;">{code(exp_dir_str)}</small>"""
    return info


def get_action_info(state: Info):
    step_info = state.exp_result.steps_info[state.step]
    action_info = f"""\
**Action:**

{code(step_info.action)}
"""
    think = step_info.agent_info.get("think", None)
    if think is not None:
        action_info += f"""
**Think:**

{code(think)}"""
    return action_info


def get_state_error(state: Info):
    try:
        step_info = state.exp_result.steps_info[state.step + 1]
        err_msg = step_info.obs.get("last_action_error", None)
    except IndexError:
        err_msg = None

    if err_msg is None or len(err_msg) == 0:
        err_msg = "No Error"
    return f"""\
**Step error after action:**

{code(err_msg)}"""


def get_seeds(result_df: pd.DataFrame, task_name: str):
    str_list = []
    seed_list = []
    result_df = result_df.reset_index(inplace=False)
    for index, row in result_df[result_df["env_args.task_name"] == task_name].iterrows():
        seed = row["env_args.task_seed"]
        reward = row["cum_reward"]
        has_err = "(Task Error)" if row["err_msg"] is not None else ""
        n_step = row["n_steps"]
        str_list.append(f"seed: {seed}, reward: {reward} in {n_step} steps. {has_err}")
        seed_list.append(seed)
    return str_list, seed_list


def on_select_task(evt: gr.SelectData, df: pd.DataFrame):
    return df.iloc[evt.index[0]]["env_args.task_name"]


def update_seeds(task_name):
    global info
    seed_strings, seeds = get_seeds(info.result_df, task_name)
    return gr.Dropdown(
        choices=list(zip(seed_strings, seeds)),
        value=seeds[0],
        label=None,
        show_label=False,
        container=False,
    )


def on_select_seed(seed, taks_name):
    return EpisodeId(task_name=taks_name, seed=seed)


def new_episode(episode_id: EpisodeId, progress=gr.Progress()):
    print("new_episode", episode_id)
    global info
    info.update_exp_result(episode_id=episode_id)
    return generate_profiling(progress.tqdm), StepId(episode_id, info.step)


def fig_to_pil(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    img_pil = Image.open(buf)
    plt.close(fig)
    img_pil.save("profiling.png")
    return img_pil


def new_exp_dir(exp_dir, progress=gr.Progress()):
    global info

    if len(exp_dir) == 0:
        info.exp_list_dir = None
        return

    info.exp_list_dir = info.results_dir / exp_dir
    info.result_df = inspect_results.load_result_df(info.exp_list_dir, progress_fn=progress.tqdm)

    info.tasks_df = inspect_results.reduce_episodes(info.result_df).reset_index()

    # task name of first element
    task_name = info.tasks_df.iloc[0]["env_args.task_name"]

    return info.tasks_df, task_name


def get_directory_contents(results_dir: Path):
    return sorted([str(file.name) for file in results_dir.iterdir() if file.is_dir()], reverse=True)


def most_recent_folder(results_dir: Path):
    return inspect_results.get_most_recent_folder(results_dir).name


def refresh_exp_dir_choices(exp_dir_choice):
    global info
    return gr.Dropdown(
        choices=get_directory_contents(info.results_dir), value=exp_dir_choice, scale=1
    )


def generate_profiling(progress_fn):
    global info

    if info.exp_result is None:
        return None

    fig, ax = plt.subplots(figsize=(20, 3))
    step_times = plot_profiling(
        ax, info.exp_result.steps_info, info.exp_result.summary_info, progress_fn
    )
    fig.tight_layout()
    info.click_mapper = ClickMapper(ax, step_times=step_times)

    return fig_to_pil(fig)


def add_patch(ax, start, stop, color, label, edge=False):
    if edge:
        ax.add_patch(
            patches.Rectangle(
                (start, 0),
                stop - start,
                1,
                edgecolor=color,
                alpha=1,
                label=label,
                fill=False,
                linewidth=1,
            )
        )
    else:
        ax.add_patch(
            patches.Rectangle((start, 0), stop - start, 1, color=color, alpha=1, label=label)
        )


def plot_profiling(ax, step_info_list: list[StepInfo], summary_info: dict, progress_fn):

    # this allows to pop labels to make sure we don't use more than 1 for the legend
    labels = ["reset", "env", "agent", "exec action", "action error"]
    labels = {e: e for e in labels}

    colors = plt.get_cmap("tab20c").colors

    t0 = step_info_list[0].profiling.env_start
    all_times = []
    step_times = []
    for i, step_info in progress_fn(enumerate(step_info_list)):
        step = step_info.step

        prof = deepcopy(step_info.profiling)
        # remove t0 from elements in profiling using for
        for key, value in prof.__dict__.items():
            if isinstance(value, float):
                setattr(prof, key, value - t0)
                all_times.append(value - t0)

        if i == 0:
            # reset
            add_patch(ax, prof.env_start, prof.env_stop, colors[14], labels.pop("reset", None))

        else:
            # env
            add_patch(ax, prof.env_start, prof.env_stop, colors[1], labels.pop("env", None))

            # action
            label = labels.pop("exec action", None)
            add_patch(ax, prof.action_exec_start, prof.action_exec_stop, colors[3], label)

            if i + 1 < len(step_info_list) and step_info_list[i + 1].obs["last_action_error"]:
                # add a hollow rectangle for error
                label = labels.pop("action error", None)
                add_patch(
                    ax, prof.action_exec_start, prof.action_exec_stop, "red", label, edge=True
                )

        if step_info.action is not None:
            # Blue rectangle for agent_start to agent_stop
            add_patch(ax, prof.agent_start, prof.agent_stop, colors[10], labels.pop("agent", None))

            # Black vertical bar at agent stop
            ax.axvline(prof.agent_stop, color="black", linewidth=3)
            step_times.append(prof.agent_stop)

            ax.text(
                prof.agent_stop,
                0,
                str(step + 1),
                color="white",
                fontsize=12,
                verticalalignment="bottom",
                horizontalalignment="left",
                rotation=0,
                clip_on=True,
                antialiased=True,
                fontweight=1000,
                backgroundcolor=colors[12],
            )

        if step_info.truncated or step_info.terminated:
            if step_info.truncated:
                color = "black"
            elif step_info.terminated:
                if summary_info["cum_reward"] > 0:
                    color = "limegreen"
                else:
                    color = "black"

            ax.axvline(prof.env_stop, color=color, linewidth=4, linestyle=":")

            text = f"R:{summary_info['cum_reward']:.1f}"

            if summary_info["err_msg"]:
                text = "Err"
                color = "red"

            ax.text(
                prof.env_stop,
                0.98,
                text,
                color="white",
                fontsize=12,
                verticalalignment="top",
                horizontalalignment="right",
                rotation=0,
                clip_on=True,
                antialiased=True,
                fontweight=1000,
                backgroundcolor=color,
            )

    ax.set_ylim(0, 1)
    ax.set_xlim(0, max(all_times) + 1)
    # plt.gca().autoscale()

    ax.set_xlabel("Time")
    ax.set_yticks([])

    # position legend above outside the fig in one row
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        ncol=5,
        frameon=True,
    )

    return step_times


if __name__ == "__main__":
    run_gradio(RESULTS_DIR)
