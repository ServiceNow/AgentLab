from logging import warning
from playwright.sync_api import Page 

from PIL import Image, ImageDraw
from logging import warning
from playwright.sync_api import Page 


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

def zoom_webpage(page: Page, zoom_factor: float = 1.5):
    """
    Zooms the webpage to the specified zoom factor.

    Args:
        page: The Playwright Page object.
        zoom_factor: The zoom factor to apply (default is 1.0, which means no zoom).
    """

    if zoom_factor <= 0:
        raise ValueError("Zoom factor must be greater than 0.")

    page.evaluate(f"document.documentElement.style.zoom='{zoom_factor*100}%'")
    return page
