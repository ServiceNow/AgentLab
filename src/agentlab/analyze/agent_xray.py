import argparse
import os
import argparse
import re
import gradio as gr
import json
import pandas as pd
import json
import os

from pathlib import Path
from PIL import ImageDraw
from pathlib import Path
from PIL import Image
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from textwrap import dedent
from PIL import Image
from agentlab.analyze import inspect_results

# from browsergym.experiments.loop import ExpArgs, StepInfo
from agentlab.llm.llm_utils import count_tokens
from browsergym.experiments.loop import ExpArgs, get_exp_result, StepInfo


# -------------------------
# Main Gradio Function
# -------------------------
def run_gradio(savedir_base):
    """
    Run Gradio on the selected experiments saved at savedir_base.
    """
    global row_episode_step_ids

    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Render the blocks
        # -----------------
        # Hidden cell to store information about the experiment
        row_episode_step_ids = {}

        # 1. Render Title
        gr.Markdown(
            f"""# MiniWob - UI-Copilot Analysis

            ## 1. Choose an Experiment from the Table below.
            """
        )

        # Define the directory text boox
        savedir_base_text = gr.Textbox(
            value=savedir_base,
            visible=True,
            label="Select Experiment Directory",
            scale=4,
        )
        savedir_base_text_button = gr.Button(
            value="Update Directory",
            visible=True,
        )
        savedir_base_text_label = gr.Label(value="Invalid Directory", visible=False)

        # 2. Render DataFrame for the experiments
        _, gradio_result_df = update_gradio_df_from_savedir_base(savedir_base)
        exp_dict_gr = gr.DataFrame(gradio_result_df, height=200)

        # 3. Render Key Info about the experiment
        with gr.Accordion(label="Main Information"):
            main_info_gr = gr.Markdown(label="Experiment Hyperparameters")

            # 4. Select the episode number and episode step
            episode_gr = gr.Slider(step=0, visible=False)
            step_gr = gr.Slider(step=0, visible=False)

            # 5. Render the episode step info
            with gr.Row():
                step_info_gr = gr.Markdown(visible=False)
                action_info_gr = gr.Text(visible=False, label="Action")
                error_info_gr = gr.Textbox(visible=False, label="Next Step error")

        # 6. Render the before and after images
        with gr.Tab("Images"):
            gr_selected_bid = gr.Textbox(
                visible=False,
                label="Write a set of actions as comma separated in this format\n"
                "'T:(x,y)' or 'T:(x1, y1, x2, y2)'\n"
                "where T can be 'dot', 'line', 'rect', and the rest are coodinates.",
                scale=4,
            )
            gr_selected_bid_button = gr.Button(
                value="Overlay Action on Source Image", visible=False
            )
            with gr.Row():
                image_src = gr.Image(show_label=False, visible=False)
                image_tgt = gr.Image(show_label=False, visible=False)

        # 7. Render the HTML
        with gr.Tab("HTML"):
            html_gr = gr.Code(lines=50, interactive=False, language="html")

        # 8. Render the Pruned HTML
        with gr.Tab("Pruned HTML"):
            pruned_html_gr = gr.Code(lines=50, interactive=False, language="html", show_label=False)

        # 9. Render the Accessibility Tree
        with gr.Tab("Acc Tree"):
            acc_gr = gr.Textbox(lines=50, show_label=False)

        # 10. Render the Agent Info
        with gr.Tab("Agent Info"):
            agent_gr = gr.Textbox(
                label="Agent Info",
                visible=False,
                lines=30,
                # lines=50,
            )

        with gr.Tab("Messages"):
            messages_gr = gr.Textbox(
                label="Messagess",
                visible=False,
                lines=30,
                # lines=50,
            )

        with gr.Tab("Re-Prompt"):
            with gr.Row():
                # Create first pane for text box with original messages
                original_gr = gr.Textbox(
                    label="Original",
                    visible=False,
                    lines=30,
                    # lines=50,
                )
                # Create second pane for text box with editable messages
                editable_gr = gr.Textbox(label="New Prompt", visible=False, lines=30)
                with gr.Column():
                    # Submit button to Input to the LLM
                    submit_prompt_gr = gr.Button(value="Submit New Prompt", visible=False)
                    # Output the raw output from the LLM
                    output_gr = gr.Textbox(label="Agent Output", lines=30, visible=False)

        # TODO: FIX ENVIRONMENT
        # with gr.Tab("Environment"):
        #     # Spawn the environment to the current step (add either an image for headless=True or use chrome browser)
        #     # A submit button to submit the raw action to the environment given in a textbox (default is the current action)
        #     with gr.Row():
        #         with gr.Column():
        #             env_image_src = gr.Image(label="Image", visible=False)
        #             with gr.Row():
        #                 env_action_text = gr.Text(label="Action")
        #                 env_submit_text = gr.Button(value="Submit Action")
        #         with gr.Column():
        #             env_image_tgt = gr.Image(
        #                 visible=False,
        #                 label="Image",
        #             )
        #     with gr.Row():
        #         run_button = gr.Button(value="Run Environment in this Interface", visible=True)
        #         run_button_browser = gr.Button(value="Run Environment on New Browser", visible=True)

        with gr.Tab("Task Error"):
            task_error_gr = gr.Textbox(show_label=False, lines=50)

        with gr.Tab("Task Logs"):
            task_logs_gr = gr.Textbox(show_label=False, lines=50)

        steps_out_list = [
            step_info_gr,
            action_info_gr,
            error_info_gr,
            image_src,
            image_tgt,
            html_gr,
            pruned_html_gr,
            acc_gr,
            agent_gr,
            messages_gr,
            gr_selected_bid,
            gr_selected_bid_button,
            original_gr,
            editable_gr,
            submit_prompt_gr,
            output_gr,
            task_error_gr,
            task_logs_gr,
        ]
        # Experiment Change Callback
        # ---------------------------------
        exp_dict_gr.select(
            fn=on_select_df,
            inputs=[exp_dict_gr],
            outputs=[main_info_gr, episode_gr, step_gr] + steps_out_list,
        )

        # Episode Change Callback
        # ---------------------------------
        episode_gr.change(
            fn=on_change_episode_info,
            inputs=[episode_gr],
            outputs=[step_gr] + steps_out_list,
        )

        # Step Change Callback
        # ---------------------------------
        # Add callback to step change
        step_gr.change(
            fn=on_change_step_info,
            inputs=[step_gr],
            outputs=steps_out_list,
        )

        # Overlay on Source Image Callback
        # ---------------------------------
        def on_change_overlay_source_image(action_text, row_episode_step_ids):
            """
            Update the source image by adding the action type and coordinates
            """
            row, episode_series, step_obj = get_row_info(
                row_episode_step_ids["row_id"],
                row_episode_step_ids["episode_id"],
                row_episode_step_ids["step_id"],
            )
            image_src = exp_result.screenshots[row_episode_step_ids["step_id"]]

            action_dict_list = convert_action_text_to_dict(action_text)
            image = get_image_with_bid(
                step_obj.action,
                step_obj.obs["axtree_txt"],
                image_src,
                action_dict_list=action_dict_list,
            )
            return gr.update(value=image, visible=True)

        # Add callback to step change
        gr_selected_bid_button.click(
            fn=on_change_overlay_source_image,
            inputs=[gr_selected_bid],
            outputs=[image_src],
        )

        # Prompt Change Callback
        # ---------------------------------
        def on_change_prompt(editable_gr):
            """
            Get the output from the LLM using the prompt
            """
            row, episode_series, step_obj = get_row_info(
                row_episode_step_ids["row_id"],
                row_episode_step_ids["episode_id"],
                row_episode_step_ids["step_id"],
            )

            exp_args = exp_result.exp_args  # type: ExpArgs
            # TODO: start servers if we use custom LLMs
            chat_model = exp_args.agent_args.chat_model_args.make_chat_model()

            text = editable_gr

            # Use re.split() to split the text based on the pattern.
            messages = re.split(r"Message \d+:\n----------", text)

            if len(messages) <= 1:
                messages = [HumanMessage(content=text)]
            else:
                # TODO: bring info from agent instead of guessing
                message_types = [SystemMessage] + [HumanMessage, AIMessage] * int(len(messages) / 2)
                messages = [
                    message_types[i](content=message.strip()) for i, message in enumerate(messages)
                ]

            answer = chat_model(messages)
            return answer.content

        # Add callback to prompt change
        submit_prompt_gr.click(
            fn=on_change_prompt,
            inputs=[editable_gr],
            outputs=[output_gr],
        )

        # Environment Change Callback
        # TODO: FIX ENVIRONMENT
        # --------------------------------
        # def on_env_run(row_episode_step_ids, headless=True):
        #     """
        #     Run the environment until step step_id
        #     """
        #     if headless == "False":
        #         headless = False

        #     row, episode_series, step_obj = get_row_info(
        #         row_episode_step_ids["row_id"],
        #         row_episode_step_ids["episode_id"],
        #         row_episode_step_ids["step_id"],
        #     )

        #     task_name = f"browsergym/{episode_series['task_name']}"
        #     env = gym.make(task_name, headless=headless)
        #     obs, env_info = env.reset(seed=int(episode_series["task_seed"]))
        #     tgt_image = obs["image"]
        #     steps_list = exp_result.steps_info
        #     action_list = [s.action for s in steps_list]
        #     for step, action in enumerate(action_list):
        #         src_image = tgt_image
        #         obs_tuple = env.step(action)
        #         obs = obs_tuple[0]
        #         tgt_image = Image.fromarray(obs["image"])

        #         if step == row_episode_step_ids["step_id"]:
        #             break
        #     row_episode_step_ids["env"] = env
        #     row_episode_step_ids_update = gr.update(
        #         value=row_episode_step_ids,
        #         visible=False,
        #     )

        #     return (
        #         gr.update(label=f"Step {step}", value=src_image, visible=True),
        #         gr.update(label=f"Output", value=tgt_image, visible=True),
        #         gr.update(value=action, visible=True),
        #         row_episode_step_ids_update,
        #     )

        # run_button.click(
        #     fn=on_env_run,
        #     inputs=[row_episode_step_ids],
        #     outputs=[env_image_src, env_image_tgt, env_action_text, row_episode_step_ids],
        # )

        # run_button_browser.click(
        #     fn=on_env_run,
        #     inputs=[
        #         row_episode_step_ids,
        #         episode_gr,
        #         step_gr,
        #         gr.Label("False", visible=False),
        #     ],
        #     outputs=[env_image_src, env_image_tgt, env_action_text, row_episode_step_ids],
        # )

        def on_action_env_run(row_episode_step_ids, env_image_tgt, env_action_text):
            """
            Run the environment with the given action
            """

            step = row_episode_step_ids["step_id"]
            if isinstance(row_episode_step_ids.get("env", ""), str):
                src_image = tgt_image = env_image_tgt
                row_episode_step_ids_update = gr.update(
                    value=row_episode_step_ids,
                    visible=False,
                )
                return (
                    row_episode_step_ids_update,
                    gr.update(label=f"Step {step}", value=src_image, visible=True),
                    gr.update(label=f"Output", value=tgt_image, visible=True),
                )

            src_image = env_image_tgt

            # TODO fix string env
            assert not isinstance(env, str), "env is not initialized properly"
            obs_tuple = env.step(env_action_text)
            obs = obs_tuple[0]
            tgt_image = Image.fromarray(obs["image"])

            row_episode_step_ids_update = gr.update(
                value=row_episode_step_ids,
                visible=False,
            )

            return (
                row_episode_step_ids_update,
                gr.update(label=f"Step {step}", value=src_image, visible=True),
                gr.update(label=f"Output", value=tgt_image, visible=True),
            )

        # env_submit_text.click(
        #     fn=on_action_env_run,
        #     inputs=[row_episode_step_ids, env_image_tgt, env_action_text],
        #     outputs=[row_episode_step_ids, env_image_src, env_image_tgt],
        # )

        # Savedir Base Change Callback
        # savedir_base_text_button.click(
        #     fn=on_change_savedir_base,
        #     inputs=[savedir_base_text],
        #     outputs=(
        #         [exp_dict_gr, savedir_base_text_label]
        #         + [main_info_gr, episode_gr, step_gr]
        #         + [row_episode_step_ids, step_gr]
        #         + steps_out_list
        #     ),
        # )

    demo.queue()
    demo.launch(server_port=7887)


# -------------------------
# Public Helper Functions
# -------------------------
def on_change_episode_info(episode_id):
    """
    Update the step information when selecting a different step
    """

    # update_episode_id
    row_id = row_episode_step_ids["row_id"]
    step_id = 0

    row_episode_step_ids["step_id"] = step_id
    row_episode_step_ids["episode_id"] = episode_id

    _, episode_series, step_obj, exp_result = get_row_info(row_id, episode_id, step_id)

    step_max = len(exp_result.steps_info) - 1

    step_info_gr_update = gr.update(
        value=step_id,
        minimum=0,
        maximum=step_max,
        step=0,
        label="Step",
        visible=True,
    )
    return [step_info_gr_update] + update_step_info(row_id, episode_id, step_id)


def on_change_step_info(step_id):
    """
    Update the step information when selecting a different step
    """
    return update_step_info(
        row_episode_step_ids["row_id"], row_episode_step_ids["episode_id"], step_id
    )


# extract action_text into a list of dict using type:coordinates
def convert_action_text_to_dict(action_text):
    """
    convert action_text into a list of dict using type:coordinates
    """
    action_dict_list = []
    for action_text_i in action_text.split("),"):
        action_text_i += ")"
        action_text_i = action_text_i.strip()
        if action_text_i == "":
            continue
        if len(action_text_i.split(":")) != 2:
            continue
        try:
            action_type = action_text_i.split(":")[0]
            coordinates = action_text_i.split(":")[1]
            coordinates = coordinates.replace("(", "").replace(")", "").split(",")
            coordinates = [float(c) for c in coordinates]
        except:
            continue
        # add to action_dict_list
        if action_type == "dot":
            action_dict_list += [{"x": coordinates[0], "y": coordinates[1], "type": "dot"}]

        elif action_type == "line":
            action_dict_list += [
                {
                    "x1": coordinates[0],
                    "y1": coordinates[1],
                    "x2": coordinates[2],
                    "y2": coordinates[3],
                    "type": "line",
                }
            ]
        elif action_type == "rect":
            action_dict_list += [
                {
                    "x1": coordinates[0],
                    "y1": coordinates[1],
                    "width": coordinates[2],
                    "height": coordinates[3],
                    "type": "rect",
                }
            ]

    return action_dict_list


# -------------------------
# Private Helper Functions
# -------------------------
def on_change_savedir_base(savedir_base_text):
    """
    Update the dataframe based on the savedir_base
    """
    global gradio_result_df
    global results_df

    if os.path.exists(savedir_base_text) and os.path.isdir(savedir_base_text):
        results_df, gradio_result_df = update_gradio_df_from_savedir_base(savedir_base_text)
        return [gr.update(value=gradio_result_df, visible=True), gr.update(visible=False)] + [
            gr.update(visible=False)
        ] * 23
    else:
        return [gr.update(visible=False), gr.update(visible=True)] + [gr.update(visible=False)] * 23


# Dataframe Change Callback
# ---------------------------------
def get_row_info(row_id, episode_id, step_id):
    """
    Get the row, episode series and step object
    """
    row = from_gradio_id_to_result_df_subset(row_id)

    if row.ndim == 1:
        episode_series = row
    else:
        episode_series = row.reset_index().iloc[episode_id]

    exp_result = get_exp_result(savedir_base / Path(episode_series["exp_dir"].name))
    try:
        step_list = exp_result.steps_info

    except FileNotFoundError:
        step_list = []
    if step_id >= len(step_list) or step_id < 0:
        step_obj = None
    else:
        step_obj = step_list[step_id]

    return row, episode_series, step_obj, exp_result


def on_select_df(evt: gr.SelectData, df):
    """
    Update the main information when selecting a different experiment
    """
    global row_episode_step_ids

    # start with zero episode and step as default
    row_id = evt.index[0]
    episode_id = 0
    step_id = 0

    row, episode_series, step_obj, exp_result = get_row_info(row_id, episode_id, step_id)

    # get max
    step_max = len(exp_result.steps_info) - 1

    # get reward info
    avg_reward = row.cum_reward.mean()

    cum_rewards = (
        ", ".join([f"{r:.1f}" for r in row.cum_reward]) if row.ndim != 1 else str(row.cum_reward)
    )

    # Main information (not step depedent)
    agent_name = exp_result.exp_args.agent_args.agent_name
    task_name = episode_series["env_args.task_name"]
    episode_max = 1 if row.ndim == 1 else len(row)

    row_episode_step_ids["row_id"] = row_id
    row_episode_step_ids["episode_id"] = episode_id
    row_episode_step_ids["step_id"] = step_id

    main_info_list = [
        # row_episode_step_ids,
        gr.update(
            value=dedent(
                f"""\
            ## Experiment id: {row_id}

            Agent `{agent_name}` on Task `{task_name}`

            **Cumulative Rewards**: {avg_reward:.1f} ({cum_rewards})

            """
            ),
            visible=True,
        ),
        gr.update(
            value=episode_id,
            minimum=0,
            maximum=episode_max - 1,
            step=0,
            label="Episode",
            visible=True,
        ),
        gr.update(
            value=step_id,
            minimum=0,
            maximum=step_max,
            step=0,
            label="Step",
            visible=True,
        ),
    ]

    # Aux information that is step dependent
    step_info_list = update_step_info(row_id, episode_id, step_id)

    return main_info_list + step_info_list


def update_step_info(row_id, episode_id, step_id):
    """
    Update the step information when selecting a different step
    """
    # update row_episode_step_ids
    row_episode_step_ids["row_id"] = row_id
    row_episode_step_ids["episode_id"] = episode_id
    row_episode_step_ids["step_id"] = step_id

    row, episode_series, step_obj, exp_result = get_row_info(row_id, episode_id, step_id)

    _, _, next_step_obj, _ = get_row_info(row_id, episode_id, step_id + 1)

    step_obj = step_obj  # type: StepInfo

    step_max = len(exp_result.steps_info) - 1

    # Step Info
    cumulative_reward = episode_series["cum_reward"]

    if episode_series["err_msg"] is None:
        task_err_msg = "No Error"
    else:
        task_err_msg = f"{episode_series['err_msg']}\n\n{episode_series['stack_trace']}"

    try:
        task_logs = exp_result.logs
    except FileNotFoundError:
        task_logs = ""

    obs = step_obj.obs if step_obj is not None else None
    next_obs = next_step_obj.obs if next_step_obj is not None else None
    if obs is None:
        goal = "No Goal"
    else:
        goal = obs["goal"]

    step_info = dedent(
        f"""\
### Step {step_id} / {step_max}

**Goal:** {goal}

**Cumulative Reward:** {cumulative_reward}

**exp_dir:**
<small>{episode_series['exp_dir'].parent.name}/{episode_series['exp_dir'].name}</small>"""
    )

    # Action Info
    if step_obj is None:
        action = None
    else:
        action = step_obj.action
    if action is None or action == "":
        action_info = "No Action"
    else:
        action_info = convert_action_dict_to_markdown(action)

    if step_obj is not None:
        if "think" in step_obj.agent_info:
            action_info += "\n\n<think>\n" + step_obj.agent_info["think"] + "\n</think>"

    # Error Logs
    if next_obs is None:
        error_info = ""
    else:
        error_info = next_obs["last_action_error"]
    if error_info == "":
        error_info = "## No Error Logs"

    screenshots = exp_result.screenshots
    # back node id
    # extract
    if step_id + 1 < len(screenshots):
        image_tgt = screenshots[step_id + 1]

    else:
        image_tgt = None

    if step_obj is None:
        agent_info = None
    else:
        agent_info = step_obj.agent_info

    # Add messages related to the agent
    if agent_info is None:
        messages = "## No Agent Info"
    else:
        messages = agent_info.get("chat_messages", "")

    if isinstance(messages, str):
        messages_md = messages
    if isinstance(messages, (list, tuple)):
        messages_md = ""
        for i, m in enumerate(messages):
            if isinstance(m, list):
                m = "\n".join([part["text"] for part in m if part["type"] == "text"])
            messages_md += f"Message {i}:\n---------- \n\n{m}\n\n"

    # convert dict of agent_info to markdown
    agent_info_dict = agent_info
    if step_obj is None or agent_info_dict is None:
        agent_info = "## No Agent Info"
    else:
        # convert dict to markdown
        agent_info = convert_to_markdown(agent_info_dict)

    # Images
    if obs is None or step_obj is None or "axtree_txt" not in obs:
        image_src = None
        action_bid_list = ""
        html = ""
        pruned_html = ""
        acc_tree = ""
    else:
        exp_args = exp_result.exp_args  # type: ExpArgs
        model_name = exp_args.agent_args.chat_model_args.model_name

        image_src_org = screenshots[step_id]
        image_src = get_image_with_bid(step_obj.action, obs["axtree_txt"], image_src_org)
        action_bid_list = get_action_bid(step_obj.action, obs["axtree_txt"])
        html = _add_token_count(obs["dom_txt"], model_name)
        pruned_html = _add_token_count(obs["pruned_html"], model_name)
        acc_tree = _add_token_count(obs["axtree_txt"], model_name)

    return [
        # goal
        gr.update(value=step_info, visible=True),
        # action info
        gr.update(value=action_info, visible=True),
        # error logs
        gr.update(value=error_info, visible=True),
        # images
        gr.update(value=image_src, visible=True),
        gr.update(value=image_tgt, visible=True),
        # # html, pruned html, acc tree, agent info, messages
        gr.update(value=html, visible=True),
        gr.update(value=pruned_html, visible=True),
        gr.update(value=acc_tree, visible=True),
        gr.update(value=agent_info, visible=True),
        gr.update(value=messages_md, visible=True),
        # # overlay actions
        gr.update(visible=True, value=get_text_from_action_list(action_bid_list)),
        gr.update(visible=True, value="Submit"),
        # # messages for re-prompt
        gr.update(value=messages_md, visible=True),
        gr.update(value=messages_md, visible=True),
        gr.update(visible=True),
        gr.update(visible=True, value=""),
        gr.update(visible=True, value=task_err_msg),
        gr.update(visible=True, value=task_logs),
    ]


def get_text_from_action_list(action_bid_list):
    """
    Get the text from the action bid list
    """
    action_text = ""

    if action_bid_list is None or action_bid_list == "" or len(action_bid_list) == 0:
        return action_text

    for action_bid in action_bid_list:
        if action_bid["type"] == "dot":
            action_text += f"dot:({action_bid['x']}, {action_bid['y']}), "
        elif action_bid["type"] == "line":
            action_text += f"line:({action_bid['x1']}, {action_bid['y1']}, {action_bid['x2']}, {action_bid['y2']}), "
        elif action_bid["type"] == "rect":
            action_text += f"rect:({action_bid['x1']}, {action_bid['y1']}, {action_bid['width']}, {action_bid['height']}), "

    # remove last comma
    action_text = action_text[:-2]

    return action_text


def get_image_with_bid(action, accessibility_tree, image_src, action_dict_list=None):
    """
    Get the image with the action bid drawn on it
    """
    if action is None:
        return image_src

    if action_dict_list is None:
        action_dict_list = get_action_bid(action, accessibility_tree)

    image_new = image_src.copy()
    image_new = image_new.convert("RGBA")
    image_new.putalpha(255)
    color_list = ["red", "blue", "green"]
    if action_dict_list is None:
        return image_new

    for i, action_dict in enumerate(action_dict_list):
        color_id = i % len(color_list)
        # if type is line draw a line
        if action_dict["type"] == "dot":
            draw = ImageDraw.Draw(image_new)
            y = action_dict["y"]
            x = action_dict["x"]
            draw.ellipse(
                (x - 5, y - 5, x + 5, y + 5),
                fill=color_list[color_id],
                outline=color_list[color_id],
            )
        elif action_dict["type"] == "line":
            draw = ImageDraw.Draw(image_new)
            draw.line(
                (
                    action_dict["x1"],
                    action_dict["y1"],
                    action_dict["x2"],
                    action_dict["y2"],
                ),
                fill=color_list[color_id],
                width=5,
            )
        else:
            raise ValueError(f"Unknown type {action_dict['type']}")
            # save new image with draw
    return image_new


def _add_token_count(text, model_name, format_str="# token count: {}"):
    n_tokens = count_tokens(text, model_name)
    return format_str.format(n_tokens) + f" (tokenized by {model_name})\n\n" + text


def from_gradio_id_to_result_df_subset(row_id):
    """
    Map gradio row id to result_df subset
    """
    global result_df
    global gradio_result_df

    index_columns = result_df.index.names
    index_values = gradio_result_df.loc[row_id, index_columns].values
    result_subset = result_df.loc[tuple(index_values)]

    if result_subset.ndim == 1:
        result_subset["env_args.task_name"] = index_values[0]

    return result_subset


def get_action_bid(action, accessibility_tree):
    """
    Get the action bid from the episode
    """
    if action is None:
        return None

    # convert string to dict
    try:
        action_list = json.loads(action)
    except json.JSONDecodeError:
        return None
    # convert dict to list
    if isinstance(action_list, dict):
        action_list = [action_list]

    if len(action_list) == 0:
        return None

    else:
        # convert x, y into tuples
        action_dict_list = []
        for action_bid in action_list:
            if action_bid["action_type"] in ["hover", "type", "mouse_down", "click", "mouse_up"]:
                type_click = "dot"
                if "bid" in action_bid and "x" not in action_bid:
                    coordinates = get_bid_coordiantes(accessibility_tree, action_bid["bid"])
                    if coordinates is None:
                        continue
                    x = coordinates[0]
                    y = coordinates[1]

                else:
                    x = action_bid["x"]
                    y = action_bid["y"]

                action_dict = {"x": float(x), "y": float(y), "type": type_click}

            action_dict["action_bid"] = action_bid

            action_dict_list += [action_dict]

        return action_dict_list


def convert_to_markdown(data, level=0):
    """
    Converts a dictionary to a Markdown string
    """
    if isinstance(data, dict):
        result = ""
        for key, value in data.items():
            result += f"{'#' * (level + 1)} {key}\n\n"
            result += convert_to_markdown(value, level + 1)
        return result
    elif isinstance(data, list):
        result = ""
        for item in data:
            result += convert_to_markdown(item, level)
        return result
    elif isinstance(data, str):
        # Format the string as Markdown
        formatted_string = data.strip()
        if level > 0:
            # Add appropriate Markdown formatting for nested strings
            formatted_string = "> " * level + formatted_string
        return formatted_string + "\n\n"
    else:
        return str(data) + "\n\n"


def convert_action_dict_to_markdown(json_dict):
    """
    prettify an action json dict into a markdown using basic python functions
    """
    # return empty string if empty or None
    if json_dict is None or json_dict == {}:
        return ""

    try:
        # convert json to dict
        if isinstance(json_dict, str):
            json_dict = json.loads(json_dict)

        if isinstance(json_dict, dict):
            json_dict = [json_dict]

        if not isinstance(json_dict, (list, tuple)):
            raise ValueError(
                f"Expected json_dict to be a list of dict, list or tuple but got {json_dict}"
            )

        action_list = []
        for action in json_dict:
            action_type = action.pop("action_type")
            args = [str(action.pop(arg)) for arg in ("bid", "css_selector") if arg in action]
            kwargs = [f"{k}={v}" for k, v in action.items()]
            action_list.append(f"{action_type}  {'  '.join(args)}  {'  '.join(kwargs)}")

        return f"\n".join(action_list)
    except json.JSONDecodeError:
        return json_dict


def get_bid_coordiantes(accessibility_tree, action_bid):
    """
    Get the coordinates of the bid with name action_bid
    """
    # get the line with the bid
    lines = accessibility_tree.split("\n")
    # get the line with [action_bid]
    bid_list = [i for i, line in enumerate(lines) if f"[{action_bid}]" in line]
    if len(bid_list) == 0:
        return None
    bid_line = lines[bid_list[0]]
    # get the only tuple in there
    # Find the start and end indexes of the tuple
    start_index = bid_line.find("(")
    end_index = bid_line.find(")")

    # Extract the tuple as a substring and convert it to a tuple
    tuple_str = bid_line[start_index + 1 : end_index]  # Exclude parentheses
    coordinates = tuple(map(float, tuple_str.split(",")))

    return coordinates


def update_gradio_df_from_savedir_base(savedir_base):
    global result_df
    global gradio_result_df

    # return no results if savedir_base does not exist
    if not os.path.exists(savedir_base):
        return pd.DataFrame(), pd.DataFrame()

    result_df = inspect_results.load_result_df(savedir_base)

    # return empty dataframe if no results
    if result_df is None:
        return pd.DataFrame(), pd.DataFrame()

    gradio_result_df = inspect_results.reduce_episodes(result_df).reset_index()
    # add a first column 'id' from 0 to n for gradio_result_df
    gradio_result_df.insert(0, "id", range(len(gradio_result_df)))

    return result_df, gradio_result_df


if __name__ == "__main__":
    """
    Run Gradio on the selected experiments saved at savedir_base.
    """
    default_result_dir = str(inspect_results.get_most_recent_folder())
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-sb",
        "--savedir_base",
        default=None,
        help="folder where experiments are saved",
    )

    args, unknown = parser.parse_known_args()
    if args.savedir_base is None:
        savedir_base = default_result_dir
    else:
        savedir_base = args.savedir_base

    savedir_base = Path(savedir_base)

    run_gradio(savedir_base)
