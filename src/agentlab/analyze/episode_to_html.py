import base64
from io import BytesIO
from pathlib import Path

from agentlab.experiments.loop import ExpResult
from agentlab.experiments.study import get_most_recent_study
from agentlab.llm.llm_utils import BaseMessage as AgentLabBaseMessage


def exp_result_to_html(
    exp_result: ExpResult,
    steps_open: bool = True,
    som_open: bool = False,
    axtree_open: bool = False,
    html_open: bool = False,
    prompt_open: bool = False,
    embed_images: bool = True,
) -> str:
    """
    Convert an ExpResult to HTML with collapsible sections.

    Args:
        exp_result: ExpResult object containing experiment data
        steps_open: Whether step sections start expanded (default: True)
        som_open: Whether SOM screenshot sections start expanded (default: False)
        axtree_open: Whether AXTree sections start expanded (default: False)
        html_open: Whether HTML sections start expanded (default: False)
        prompt_open: Whether Prompt sections start expanded (default: False)
        embed_images: Whether to embed images as base64 or use file paths (default: True)

    Returns:
        str: HTML string with collapsible episode visualization
    """
    # Get basic episode info
    env_args = exp_result.exp_args.env_args
    steps_info = exp_result.steps_info

    # Build HTML structure
    html_parts = []

    # Add CSS for styling (unchanged)
    html_parts.append(
        """
    <style>
        .episode-container {
            font-family: monospace;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }
        .episode-title {
            font-size: 24px;
            font-weight: bold;
            margin-bottom: 10px;
            color: #333;
        }
        .episode-meta {
            font-size: 14px;
            color: #666;
            margin-bottom: 20px;
        }
        details {
            margin: 5px 0;
            border: 1px solid #ddd;
            border-radius: 4px;
        }
        details[open] {
            background-color: #f9f9f9;
        }
        summary {
            padding: 10px;
            cursor: pointer;
            font-weight: bold;
            background-color: #f0f0f0;
            border-bottom: 1px solid #ddd;
        }
        summary:hover {
            background-color: #e0e0e0;
        }
        .content {
            padding: 15px;
        }
        .step-content {
            padding: 15px;
        }
        .screenshot {
            max-width: 100%;
            border: 1px solid #ccc;
            margin: 10px 0;
        }
        .action-text {
            background-color: #f8f8f8;
            padding: 10px;
            border-left: 4px solid #007acc;
            margin: 10px 0;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .think-text {
            background-color: #fff3cd;
            padding: 10px;
            border-left: 4px solid #ffc107;
            margin: 10px 0;
            font-family: monospace;
            white-space: pre-wrap;
        }
        .code-block {
            background-color: #f8f9fa;
            padding: 10px;
            border: 1px solid #e9ecef;
            border-radius: 4px;
            font-family: monospace;
            white-space: pre-wrap;
            max-height: 400px;
            overflow-y: auto;
        }
        .goal-section {
            background-color: #e8f5e8;
            padding: 15px;
            border-left: 4px solid #28a745;
            margin: 10px 0;
        }
        .step-separator {
            border-top: 2px solid #007acc;
            margin: 20px 0;
        }
        .nested-details {
            margin: 10px 0;
            border: 1px solid #ccc;
        }
        .nested-details summary {
            background-color: #f8f9fa;
            padding: 8px;
            font-size: 14px;
        }
        .chat-message {
            margin: 10px 0;
            padding: 10px;
            border-left: 3px solid #007acc;
            background-color: #f8f9fa;
        }
        .chat-role {
            font-weight: bold;
            color: #007acc;
            margin-bottom: 5px;
        }
    </style>
    """
    )

    # Start main container
    html_parts.append('<div class="episode-container">')

    # Episode title and metadata
    html_parts.append(f'<div class="episode-title">{env_args.task_name}</div>')
    html_parts.append(f'<div class="episode-meta">Seed: {env_args.task_seed}</div>')

    # Goal section - format like xray
    if len(steps_info) > 0:
        try:
            goal = steps_info[0].obs.get("goal_object", "No goal specified")
            goal_html = _format_goal(goal)
            html_parts.append(
                f'<div class="goal-section"><strong>Goal:</strong><br>{goal_html}</div>'
            )
        except (IndexError, AttributeError):
            pass

    # Process each step
    for i, step_info in enumerate(steps_info):
        if step_info.action is None and i == 0:
            continue  # Skip initial reset step if no action

        # Step container with collapsible wrapper
        step_open_attr = "open" if steps_open else ""
        html_parts.append(f"<details {step_open_attr}>")
        html_parts.append(f"<summary>Step {i}</summary>")
        html_parts.append('<div class="step-content">')

        # Screenshot (flat in step)
        screenshot_html = _get_screenshot_html(exp_result, i, embed_images)
        html_parts.append(screenshot_html)

        # Action (flat in step)
        if step_info.action is not None:
            html_parts.append(
                f'<div class="action-text"><strong>Action:</strong><br>{_escape_html(step_info.action)}</div>'
            )

        # Think (flat in step)
        think_content = step_info.agent_info.get("think", "")
        if think_content:
            html_parts.append(
                f'<div class="think-text"><strong>Think:</strong><br>{_escape_html(think_content)}</div>'
            )

        # SOM Screenshot (nested collapsible)
        som_screenshot_html = _get_som_screenshot_html(exp_result, i, som_open, embed_images)
        if som_screenshot_html:
            html_parts.append(som_screenshot_html)

        # AXTree (nested collapsible)
        axtree_content = step_info.obs.get("axtree_txt", "")
        if axtree_content:
            axtree_open_attr = "open" if axtree_open else ""
            html_parts.append(
                f"""
            <details class="nested-details" {axtree_open_attr}>
                <summary>AXTree</summary>
                <div class="content">
                    <div class="code-block">{_escape_html(axtree_content)}</div>
                </div>
            </details>
            """
            )

        # HTML (nested collapsible)
        html_content = step_info.obs.get("dom_txt", "")
        if html_content:
            html_open_attr = "open" if html_open else ""
            html_parts.append(
                f"""
            <details class="nested-details" {html_open_attr}>
                <summary>HTML</summary>
                <div class="content">
                    <div class="code-block">{_escape_html(html_content)}</div>
                </div>
            </details>
            """
            )

        # Prompt/Chat messages (nested collapsible) - format like xray
        chat_messages = step_info.agent_info.get("chat_messages", [])
        if chat_messages:
            prompt_open_attr = "open" if prompt_open else ""
            chat_html = _format_chat_messages_like_xray(chat_messages)
            html_parts.append(
                f"""
            <details class="nested-details" {prompt_open_attr}>
                <summary>Prompt</summary>
                <div class="content">
                    {chat_html}
                </div>
            </details>
            """
            )

        # Close step container
        html_parts.append("</div>")  # step-content
        html_parts.append("</details>")  # step

    # Close main container
    html_parts.append("</div>")

    return "".join(html_parts)


def _get_screenshot_html(exp_result, step: int, embed_images: bool) -> str:
    """Get HTML for main screenshot at given step."""
    try:
        if embed_images:
            screenshot = exp_result.get_screenshot(step, som=False)
            return _image_to_html(screenshot, f"Screenshot {step}")
        else:
            screenshot_path = exp_result.get_screenshot_path(step, som=False)
            return _path_to_html(screenshot_path, f"Screenshot {step}")
    except (FileNotFoundError, IndexError):
        return "<p>Screenshot not available</p>"


def _get_som_screenshot_html(exp_result, step: int, som_open: bool, embed_images: bool) -> str:
    """Get HTML for SOM screenshot if available."""
    try:
        if embed_images:
            screenshot_som = exp_result.get_screenshot(step, som=True)
            if screenshot_som:
                som_open_attr = "open" if som_open else ""
                som_html = _image_to_html(screenshot_som, f"SOM Screenshot {step}")
                return f"""
                <details class="nested-details" {som_open_attr}>
                    <summary>Screenshot_som[{step}]</summary>
                    <div class="content">
                        {som_html}
                    </div>
                </details>
                """
        else:
            screenshot_path = exp_result.get_screenshot_path(step, som=True)
            if screenshot_path and screenshot_path.exists():
                som_open_attr = "open" if som_open else ""
                som_html = _path_to_html(screenshot_path, f"SOM Screenshot {step}")
                return f"""
                <details class="nested-details" {som_open_attr}>
                    <summary>Screenshot_som[{step}]</summary>
                    <div class="content">
                        {som_html}
                    </div>
                </details>
                """
    except (FileNotFoundError, IndexError):
        pass
    return ""


def _image_to_html(image, alt_text: str) -> str:
    """Convert PIL Image to HTML img tag with base64 encoding."""
    if image is None:
        return f"<p>{alt_text} not available</p>"

    # Convert PIL Image to base64
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode()

    return f'<img src="data:image/png;base64,{img_str}" alt="{alt_text}" class="screenshot">'


def _path_to_html(image_path, alt_text: str) -> str:
    """Convert image path to HTML img tag."""
    if image_path is None or not image_path.exists():
        return f"<p>{alt_text} not available</p>"

    # Convert to absolute path and use file:// protocol
    abs_path = image_path.resolve()
    return f'<img src="file://{abs_path}" alt="{alt_text}" class="screenshot">'


# Rest of the helper functions remain unchanged...
def _format_goal(goal) -> str:
    """Format goal object like xray does - using code blocks."""
    if goal is None:
        return "<div class='code-block'>No goal specified</div>"

    # Format like xray's AgentLabBaseMessage approach
    goal_str = str(AgentLabBaseMessage("", goal))

    return f"<div class='code-block'>{_escape_html(goal_str)}</div>"


def _format_chat_messages_like_xray(messages) -> str:
    """Format chat messages like xray does - with proper role separation."""
    if not messages:
        return "<div class='code-block'>No chat messages</div>"

    formatted_parts = []

    for i, msg in enumerate(messages):
        message_html = []

        if hasattr(msg, "role") and hasattr(msg, "content"):
            # Handle BaseMessage objects
            role = getattr(msg, "role", "unknown")
            content = getattr(msg, "content", str(msg))

            message_html.append(f'<div class="chat-role">{role.upper()}</div>')
            message_html.append(f'<div class="code-block">{_escape_html(str(content))}</div>')

        elif isinstance(msg, dict):
            # Handle dict messages
            role = msg.get("role", "unknown")
            content = msg.get("content", str(msg))

            message_html.append(f'<div class="chat-role">{role.upper()}</div>')

            if isinstance(content, list):
                # Handle multi-part content like xray
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "text":
                            message_html.append(
                                f'<div class="code-block">{_escape_html(part.get("text", ""))}</div>'
                            )
                        elif part.get("type") == "image":
                            message_html.append('<div class="code-block">[IMAGE]</div>')
                        elif part.get("type") == "tool_use":
                            tool_str = _format_tool_call_like_xray(part)
                            message_html.append(
                                f'<div class="code-block">{_escape_html(tool_str)}</div>'
                            )
                        else:
                            message_html.append(
                                f'<div class="code-block">{_escape_html(str(part))}</div>'
                            )
                    else:
                        message_html.append(
                            f'<div class="code-block">{_escape_html(str(part))}</div>'
                        )
            else:
                message_html.append(f'<div class="code-block">{_escape_html(str(content))}</div>')
        else:
            # Handle other message types
            message_html.append(f'<div class="code-block">{_escape_html(str(msg))}</div>')

        formatted_parts.append(f'<div class="chat-message">{"".join(message_html)}</div>')

    return "".join(formatted_parts)


def _format_tool_call_like_xray(tool_item: dict) -> str:
    """Format tool calls like xray does."""
    name = tool_item.get("name", "unknown")
    input_data = tool_item.get("input", {})
    call_id = tool_item.get("call_id", "unknown")

    return f"Tool Call: {name} `{input_data}` (call_id: {call_id})"


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    if not isinstance(text, str):
        text = str(text)

    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#x27;")
    )


if __name__ == "__main__":

    from agentlab.experiments.exp_utils import RESULTS_DIR

    result_dir = get_most_recent_study(RESULTS_DIR, contains=None)
    for exp_dir in result_dir.iterdir():
        if exp_dir.is_dir():
            break

    print(f"Using first exp_dir in most recent study:\n{exp_dir}")
    exp_result = ExpResult(exp_dir=exp_dir)

    page = exp_result_to_html(exp_result, embed_images=False)

    output_file = exp_dir / "episode.html"
    print(f"Writing HTML to\n{output_file}")
    output_file.write_text(page)
    # cmd open output_file using subprocess
    import subprocess

    subprocess.run(["open", str(output_file)])  # macOS command to open HTML file
