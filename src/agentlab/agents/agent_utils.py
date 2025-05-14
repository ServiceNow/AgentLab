from PIL import Image, ImageDraw
from logging import warning



"""
This module contains utility functions for handling observations and actions in the context of agent interactions.
"""

def tag_screenshot_with_action(screenshot: Image, action: str) -> Image:
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
    return screenshot
