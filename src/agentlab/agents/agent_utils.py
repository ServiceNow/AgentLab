import copy

from PIL import Image, ImageDraw
from playwright.sync_api import Page

from agentlab.analyze import overlay_utils
from agentlab.llm.llm_utils import img_to_base_64


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


def overlay_action(obs, action):
    """Overlays actions on screenshot in-place"""
    act_img = copy.deepcopy(obs["screenshot"])
    act_img = Image.fromarray(act_img)

    new_obs_properties = copy.deepcopy(obs["extra_element_properties"])
    import os

    if os.getenv("AGENTLAB_USE_RETINA"):
        # HACK: divide everything by 2 in the obs
        # TODO: make this more robust by changing login in annotate_action directly (or maybe in the obs section?)
        for key, value in new_obs_properties.items():
            try:
                new_obs_properties[key]["bbox"] = [elem / 2 for elem in value["bbox"]]
            except:
                pass

    overlay_utils.annotate_action(act_img, action, properties=new_obs_properties)
    return img_to_base_64(act_img)
