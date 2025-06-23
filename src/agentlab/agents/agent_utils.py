from logging import warning
from typing import Optional, Tuple

import numpy as np
from PIL import Image, ImageDraw
from playwright.sync_api import Page

"""
This module contains utility functions for handling observations and actions in the context of agent interactions.
"""


def tag_screenshot_with_action(screenshot: Image, action: str | list[str]) -> Image:
    """
    If action is a coordinate action, try to render it on the screenshot.

    e.g. mouse_click(120, 130) -> draw a dot at (120, 130) on the screenshot

    Args:
        screenshot: The screenshot to tag.
        action: The action to tag the screenshot with.

    Returns:
        The tagged screenshot.

    Raises:
        ValueError: If the action parsing fails.
    """
    import copy
    actions = copy.deepcopy(action)  # Avoid modifying the original action
    if action is str:
        actions = [actions]
    
    for action in actions:
        if action.startswith("mouse_click"):
            try:
                coords = action[action.index("(") + 1 : action.index(")")].split(",")
                coords = [c.strip() for c in coords]
                if len(coords) not in [2, 3]:
                    raise ValueError(f"Invalid coordinate format: {coords}")
                if coords[0].startswith("x="):
                    coords[0] = coords[0][2:]
                if coords[1].startswith("y="):
                    coords[1] = coords[1][2:]
                x, y = float(coords[0].strip()), float(coords[1].strip())
                draw = ImageDraw.Draw(screenshot)
                radius = 5
                draw.ellipse(
                    (x - radius, y - radius, x + radius, y + radius), fill="blue", outline="blue"
                )
            except (ValueError, IndexError) as e:
                warning(f"Failed to parse action '{action}': {e}")

        elif action.startswith("mouse_drag_and_drop"):
            try:
                func_name, parsed_args = parse_func_call_string(action)
                if func_name == "mouse_drag_and_drop" and parsed_args is not None:
                    args, kwargs = parsed_args
                    x1, y1, x2, y2 = None, None, None, None

                    if args and len(args) >= 4:
                        # Positional arguments: mouse_drag_and_drop(x1, y1, x2, y2)
                        x1, y1, x2, y2 = map(float, args[:4])
                    elif kwargs:
                        # Keyword arguments: mouse_drag_and_drop(from_x=x1, from_y=y1, to_x=x2, to_y=y2)
                        x1 = float(kwargs.get("from_x", 0))
                        y1 = float(kwargs.get("from_y", 0))
                        x2 = float(kwargs.get("to_x", 0))
                        y2 = float(kwargs.get("to_y", 0))

                    if all(coord is not None for coord in [x1, y1, x2, y2]):
                        draw = ImageDraw.Draw(screenshot)
                        # Draw the main line
                        draw.line((x1, y1, x2, y2), fill="red", width=2)
                        # Draw arrowhead at the end point using the helper function
                        draw_arrowhead(draw, (x1, y1), (x2, y2))
            except (ValueError, IndexError) as e:
                warning(f"Failed to parse action '{action}': {e}")
    return screenshot


def add_mouse_pointer_from_action(screenshot: Image, action: str) -> Image.Image:

    if action.startswith("mouse_click"):
        try:
            coords = action[action.index("(") + 1 : action.index(")")].split(",")
            coords = [c.strip() for c in coords]
            if len(coords) not in [2, 3]:
                raise ValueError(f"Invalid coordinate format: {coords}")
            if coords[0].startswith("x="):
                coords[0] = coords[0][2:]
            if coords[1].startswith("y="):
                coords[1] = coords[1][2:]
            x, y = int(coords[0].strip()), int(coords[1].strip())
            screenshot = draw_mouse_pointer(screenshot, x, y)
        except (ValueError, IndexError) as e:
            warning(f"Failed to parse action '{action}': {e}")
    return screenshot


def draw_mouse_pointer(image: Image.Image, x: int, y: int) -> Image.Image:
    """
    Draws a semi-transparent mouse pointer at (x, y) on the image.
    Returns a new image with the pointer drawn.

    Args:
        image: The image to draw the mouse pointer on.
        x: The x coordinate for the mouse pointer.
        y: The y coordinate for the mouse pointer.

    Returns:
        A new image with the mouse pointer drawn.
    """
    pointer_size = 20  # Length of the pointer
    overlay = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay)

    # Define pointer shape (a simple arrow)
    pointer_shape = [
        (x, y),
        (x + pointer_size, y + pointer_size // 2),
        (x + pointer_size // 2, y + pointer_size // 2),
        (x + pointer_size // 2, y + pointer_size),
    ]

    draw.polygon(pointer_shape, fill=(0, 0, 0, 128))  # 50% transparent black

    return Image.alpha_composite(image.convert("RGBA"), overlay)


def draw_arrowhead(draw, start, end, arrow_length=15, arrow_angle=30):
    from math import atan2, cos, radians, sin

    angle = atan2(end[1] - start[1], end[0] - start[0])
    left = (
        end[0] - arrow_length * cos(angle - radians(arrow_angle)),
        end[1] - arrow_length * sin(angle - radians(arrow_angle)),
    )
    right = (
        end[0] - arrow_length * cos(angle + radians(arrow_angle)),
        end[1] - arrow_length * sin(angle + radians(arrow_angle)),
    )
    draw.line([end, left], fill="red", width=4)
    draw.line([end, right], fill="red", width=4)


def draw_click_indicator(image: Image.Image, x: int, y: int) -> Image.Image:
    """
    Draws a click indicator (+ shape with disconnected lines) at (x, y) on the image.
    Returns a new image with the click indicator drawn.

    Args:
        image: The image to draw the click indicator on.
        x: The x coordinate for the click indicator.
        y: The y coordinate for the click indicator.

    Returns:
        A new image with the click indicator drawn.
    """
    line_length = 10  # Length of each line segment
    gap = 4  # Gap from center point
    line_width = 2  # Thickness of lines

    overlay = image.convert("RGBA").copy()
    draw = ImageDraw.Draw(overlay)

    # Draw 4 lines forming a + shape with gaps in the center
    # Each line has a white outline and black center for visibility on any background

    # Top line
    draw.line(
        [(x, y - gap - line_length), (x, y - gap)], fill=(255, 255, 255, 200), width=line_width + 2
    )  # White outline
    draw.line(
        [(x, y - gap - line_length), (x, y - gap)], fill=(0, 0, 0, 255), width=line_width
    )  # Black center

    # Bottom line
    draw.line(
        [(x, y + gap), (x, y + gap + line_length)], fill=(255, 255, 255, 200), width=line_width + 2
    )  # White outline
    draw.line(
        [(x, y + gap), (x, y + gap + line_length)], fill=(0, 0, 0, 255), width=line_width
    )  # Black center

    # Left line
    draw.line(
        [(x - gap - line_length, y), (x - gap, y)], fill=(255, 255, 255, 200), width=line_width + 2
    )  # White outline
    draw.line(
        [(x - gap - line_length, y), (x - gap, y)], fill=(0, 0, 0, 255), width=line_width
    )  # Black center

    # Right line
    draw.line(
        [(x + gap, y), (x + gap + line_length, y)], fill=(255, 255, 255, 200), width=line_width + 2
    )  # White outline
    draw.line(
        [(x + gap, y), (x + gap + line_length, y)], fill=(0, 0, 0, 255), width=line_width
    )  # Black center

    return Image.alpha_composite(image.convert("RGBA"), overlay)


def zoom_webpage(page: Page, zoom_factor: float = 1.5):
    """
    Zooms the webpage to the specified zoom factor.

    NOTE: Click actions with bid doesn't work properly when zoomed in.

    Args:
        page: The Playwright Page object.
        zoom_factor: The zoom factor to apply (default is 1.5).

    Returns:
        Page: The modified Playwright Page object.

    Raises:
        ValueError: If zoom_factor is less than or equal to 0.
    """

    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0.")

    page.evaluate(f"document.documentElement.style.zoom='{zoom_factor*100}%'")
    return page


def parse_func_call_string(call_str: str) -> Tuple[Optional[str], Optional[Tuple[list, dict]]]:
    """
    Parse a function call string and extract the function name and arguments.

    Args:
        call_str (str): A string like "mouse_click(100, 200)" or "mouse_drag_and_drop(x=10, y=20)"

    Returns:
        Tuple (func_name, (args, kwargs)), or (None, None) if parsing fails
    """
    import ast

    try:
        tree = ast.parse(call_str.strip(), mode="eval")
        if not isinstance(tree.body, ast.Call):
            return None, None

        call_node = tree.body

        # Function name
        if isinstance(call_node.func, ast.Name):
            func_name = call_node.func.id
        else:
            return None, None

        # Positional arguments
        args = []
        for arg in call_node.args:
            try:
                args.append(ast.literal_eval(arg))
            except (ValueError, TypeError):
                return None, None

        # Keyword arguments
        kwargs = {}
        for kw in call_node.keywords:
            try:
                kwargs[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                return None, None

        return func_name, (args, kwargs)

    except (SyntaxError, ValueError, TypeError):
        return None, None
