import ast
import inspect
import math
from dataclasses import dataclass
from typing import Any, Union

import matplotlib.pyplot as plt
import PIL
from browsergym.core.action.highlevel import ACTION_SUBSETS
from PIL import Image, ImageDraw

BGYM_FUNCTION_MAP = {}
for subset in ("bid", "coord"):
    for func in ACTION_SUBSETS[subset]:
        if func not in BGYM_FUNCTION_MAP:
            BGYM_FUNCTION_MAP[func.__name__] = func


@dataclass
class ArgInfo:
    function_name: str
    name: str
    value: Any
    type: str
    start_index: int
    stop_index: int


def parse_function_calls(code_string: str) -> list[ArgInfo]:
    """
    Parse a string containing multiple function calls and return a list of ArgInfo objects
    for all arguments in all function calls.

    Args:
        code_string: String containing function calls

    Returns:
        List of ArgInfo objects containing detailed information about each argument

    Example:
        >>> code = '''
        ... mouse_click(34, 59)
        ... fill("a234", "test")
        ... '''
        >>> result = parse_function_calls(code)
        >>> # Returns list of ArgInfo objects for each argument
    """
    result = []

    try:
        # Parse the code string into an AST
        tree = ast.parse(code_string)

        # Extract all function calls
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                func_name = node.func.id

                # Check if this function exists in our module
                if func_name in BGYM_FUNCTION_MAP:
                    func = BGYM_FUNCTION_MAP[func_name]

                    # Get function signature to map positional args to parameter names
                    try:
                        sig = inspect.signature(func)
                        param_names = list(sig.parameters.keys())

                        # Process positional arguments
                        for i, arg in enumerate(node.args):
                            if i < len(param_names):
                                param_name = param_names[i]
                                value = _extract_value(arg)
                                start_idx, stop_idx = _get_node_indices(code_string, arg)

                                arg_info = ArgInfo(
                                    function_name=func_name,
                                    name=param_name,
                                    value=value,
                                    type=type(value).__name__,
                                    start_index=start_idx,
                                    stop_index=stop_idx,
                                )
                                result.append(arg_info)

                        # Process keyword arguments
                        for keyword in node.keywords:
                            value = _extract_value(keyword.value)
                            start_idx, stop_idx = _get_node_indices(
                                code_string, keyword.value, keyword
                            )

                            arg_info = ArgInfo(
                                function_name=func_name,
                                name=keyword.arg,
                                value=value,
                                type=type(value).__name__,
                                start_index=start_idx,
                                stop_index=stop_idx,
                            )
                            result.append(arg_info)

                    except Exception as e:
                        # If we can't inspect the function, skip it
                        print(f"Warning: Could not process function {func_name}: {e}")
                        continue

    except SyntaxError as e:
        print(f"Syntax error in code string: {e}")
        return []

    return result


def _extract_value(node: ast.AST) -> Any:
    """
    Extract the actual value from an AST node.

    Args:
        node: AST node representing a value

    Returns:
        The extracted Python value
    """
    if isinstance(node, ast.Constant):
        # Python 3.8+ uses ast.Constant for all literals
        return node.value
    elif isinstance(node, ast.Str):
        # Fallback for older Python versions
        return node.s
    elif isinstance(node, ast.Num):
        # Fallback for older Python versions
        return node.n
    elif isinstance(node, ast.List):
        # Handle list literals
        return [_extract_value(item) for item in node.elts]
    elif isinstance(node, ast.Name):
        # Handle variable names (return as string identifier)
        return node.id
    else:
        # For other node types, return a string representation
        return ast.unparse(node) if hasattr(ast, "unparse") else str(node)


def _get_node_indices(
    source: str, node: ast.AST, keyword_node: ast.keyword = None
) -> tuple[int, int]:
    """
    Convert AST node line/column positions to absolute character indices.

    Args:
        source: Original source code string
        node: AST node (the value)
        keyword_node: If provided, use this keyword node's position as start

    Returns:
        Tuple of (start_index, stop_index) in the source string
    """
    lines = source.splitlines(keepends=True)

    # For keyword arguments, start from the keyword name
    if keyword_node is not None:
        start_line = keyword_node.lineno
        start_col = keyword_node.col_offset
    else:
        start_line = node.lineno
        start_col = node.col_offset

    # Calculate start index
    start_index = 0
    for i in range(start_line - 1):  # lineno is 1-based
        start_index += len(lines[i])
    start_index += start_col

    # End index always comes from the value node
    if hasattr(node, "end_lineno") and hasattr(node, "end_col_offset"):
        end_index = 0
        for i in range(node.end_lineno - 1):
            end_index += len(lines[i])
        end_index += node.end_col_offset
    else:
        # Fallback estimation
        if hasattr(ast, "get_source_segment"):
            segment = ast.get_source_segment(source, node)
            end_index = start_index + len(segment) if segment else start_index + 1
        else:
            end_index = start_index + 1

    return start_index, end_index


def find_bids_and_xy_pairs(args: list[ArgInfo]) -> list[ArgInfo]:
    """
    Find bid arguments and x,y coordinate pairs from a list of ArgInfo objects.

    Args:
        args: List of ArgInfo objects from parse_function_calls

    Returns:
        List of ArgInfo objects containing:
        - Original bid arguments (unchanged)
        - Merged x,y pairs with joint names, tuple values, and combined indices

    Rules for x,y pairs:
    - Must be consecutive arguments
    - Must end with 'x' and 'y' respectively
    - Must have the same prefix (everything before 'x'/'y')
    - Merged name: prefix + "_xy"
    - Merged value: (x_value, y_value) as tuple of floats
    - Merged indices: start of x to stop of y
    """
    result = []
    i = 0

    while i < len(args):
        current_arg = args[i]

        # Check if current arg name ends with 'bid'
        if current_arg.name.endswith("bid"):
            result.append(current_arg)
            i += 1
            continue

        # Check for x,y pair
        if i + 1 < len(args) and current_arg.name.endswith("x") and args[i + 1].name.endswith("y"):

            next_arg = args[i + 1]

            # Extract prefixes (everything before 'x' and 'y')
            current_prefix = current_arg.name[:-1]  # Remove 'x'
            next_prefix = next_arg.name[:-1]  # Remove 'y'

            # Check if they have the same prefix and are from the same function
            if (
                current_prefix == next_prefix
                and current_arg.function_name == next_arg.function_name
            ):

                # Create merged ArgInfo for x,y pair
                merged_name = f"{current_prefix}xy"

                # Convert values to floats and create tuple
                try:
                    x_val = float(current_arg.value)
                    y_val = float(next_arg.value)
                    merged_value = (x_val, y_val)
                except (ValueError, TypeError):
                    # If conversion fails, keep original values
                    merged_value = (current_arg.value, next_arg.value)

                merged_arg = ArgInfo(
                    function_name=current_arg.function_name,
                    name=merged_name,
                    value=merged_value,
                    type="tuple",
                    start_index=current_arg.start_index,
                    stop_index=next_arg.stop_index,
                )

                result.append(merged_arg)
                i += 2  # Skip both x and y args
                continue

        # If no special handling, skip this argument
        i += 1

    return result


def overlay_cross(
    img: Image.Image,
    coord: tuple[float, float],
    color: Union[str, tuple[int, int, int]] = "red",
    length: int = 7,
    width: int = 1,
) -> Image.Image:
    draw = ImageDraw.Draw(img)

    x, y = coord
    half_len = length // 2

    # Draw horizontal line
    draw.line([x - half_len, y, x + half_len, y], fill=color, width=width)
    # Draw vertical line
    draw.line([x, y - half_len, x, y + half_len], fill=color, width=width)

    return img


def overlay_rectangle(
    img: Image.Image,
    bbox: tuple[float, float, float, float],
    color: Union[str, tuple[int, int, int]] = "red",
    width: int = 1,
    dashed: bool = True,
) -> Image.Image:
    draw = ImageDraw.Draw(img)

    x, y, w, h = bbox

    if dashed:
        # Draw dashed rectangle
        linedashed(draw, x, y, x + w, y, color, width)
        linedashed(draw, x + w, y, x + w, y + h, color, width)
        linedashed(draw, x + w, y + h, x, y + h, color, width)
        linedashed(draw, x, y + h, x, y, color, width)
    else:
        draw.rectangle([x, y, x + w, y + h], outline=color, width=width)

    return img


# Adapted from https://stackoverflow.com/questions/51908563/dotted-or-dashed-line-with-python-pillow/58885306#58885306
def linedashed(
    draw: PIL.ImageDraw.Draw, x0, y0, x1, y1, fill, width, dash_length=4, nodash_length=8
):
    line_dx = x1 - x0  # delta x (can be negative)
    line_dy = y1 - y0  # delta y (can be negative)
    line_length = math.hypot(line_dx, line_dy)  # line length (positive)
    if line_length == 0:
        return  # Avoid division by zero in case the line length is 0
    pixel_dx = line_dx / line_length  # x add for 1px line length
    pixel_dy = line_dy / line_length  # y add for 1px line length
    dash_start = 0
    while dash_start < line_length:
        dash_end = dash_start + dash_length
        if dash_end > line_length:
            dash_end = line_length
        draw.line(
            (
                round(x0 + pixel_dx * dash_start),
                round(y0 + pixel_dy * dash_start),
                round(x0 + pixel_dx * dash_end),
                round(y0 + pixel_dy * dash_end),
            ),
            fill=fill,
            width=width,
        )
        dash_start += dash_length + nodash_length


def annotate_action(
    img: Image.Image, action_string: str, properties: dict[str, tuple], colormap: str = "tab10"
) -> str:
    """
    Annotate an image with overlays for action arguments and return colored HTML.

    Args:
        img: PIL Image to modify in place
        action_string: String containing function calls
        properties: Dict mapping bid strings to bounding boxes (x1, y1, x2, y2)
        colormap: Matplotlib colormap name for auto-color selection

    Returns:
        HTML string with arguments colored to match overlays
    """
    # Parse function calls to get all arguments
    all_args = parse_function_calls(action_string)

    # Filter to get bids and xy pairs
    filtered_args = find_bids_and_xy_pairs(all_args)

    # Get colormap
    cmap = plt.get_cmap(colormap)

    # Track colors for each filtered argument
    colors = []

    # Add overlays to image
    for i, arg_info in enumerate(filtered_args):
        # Get color from colormap
        color_rgb = cmap(i % cmap.N)
        color_255 = tuple(int(c * 255) for c in color_rgb[:3])  # Convert to 0-255 range

        colors.append(color_rgb[:3])  # Store normalized RGB for HTML

        if arg_info.name.endswith("xy"):
            # Handle x,y coordinate pairs
            x, y = arg_info.value
            overlay_cross(img, (x, y), color_255, length=9, width=3)

        elif arg_info.name.endswith("bid"):
            # Handle bid arguments with bounding boxes
            bid_value = arg_info.value
            if bid_value in properties:

                bbox = properties[bid_value]["bbox"]
                if bbox:
                    overlay_rectangle(img, bbox, color_255, width=3)

    # Generate colored HTML
    html = create_colored_html(action_string, filtered_args, colors)

    return html


def create_colored_html(action_string: str, filtered_args: list, colors: list) -> str:
    """
    Create HTML with colored arguments using start/stop indices.

    Args:
        action_string: Original action string
        filtered_args: List of ArgInfo objects with start_index/stop_index
        colors: List of RGB tuples, same length as filtered_args

    Returns:
        HTML string with colored spans
    """
    # Sort args by start position for sequential processing
    sorted_pairs = sorted(zip(filtered_args, colors), key=lambda x: x[0].start_index)

    # Build HTML with colored spans
    html_parts = []
    last_end = 0

    for arg_info, color_rgb in sorted_pairs:
        # Add uncolored text before this argument
        html_parts.append(action_string[last_end : arg_info.start_index])

        # Get the argument text
        arg_text = action_string[arg_info.start_index : arg_info.stop_index]

        # Convert color to hex
        color_hex = "#{:02x}{:02x}{:02x}".format(
            int(color_rgb[0] * 255), int(color_rgb[1] * 255), int(color_rgb[2] * 255)
        )

        # Add colored span
        html_parts.append(f'<span style="color: {color_hex}; font-weight: bold;">{arg_text}</span>')

        last_end = arg_info.stop_index

    # Add remaining text
    html_parts.append(action_string[last_end:])

    return "".join(html_parts)
