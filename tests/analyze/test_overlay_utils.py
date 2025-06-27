from PIL import Image

from agentlab.analyze import overlay_utils


def test_parse_function_calls():

    test_code = """
mouse_click(34, 59)
fill("a234", "test")
click('b123', button="right", modifiers=["Shift", "Control"])
select_option("c456", ["option1", "option2"])
"""

    result = overlay_utils.parse_function_calls(test_code)

    assert result[1].function_name == "mouse_click"
    assert result[1].name == "y"
    assert test_code[result[1].start_index : result[1].stop_index] == "59"

    assert result[8].function_name == "select_option"
    assert result[8].name == "options"
    assert test_code[result[8].start_index : result[8].stop_index] == '["option1", "option2"]'


def test_filtering_args():
    test_code = """
mouse_click(34, 59)
fill("a234", "test")
mouse_drag_and_drop(34, 59, to_x=100, to_y=200)
drag_and_drop("a123", "b456")
"""
    result = overlay_utils.parse_function_calls(test_code)
    args = overlay_utils.find_bids_and_xy_pairs(result)

    assert len(args) == 6  # Expecting 4 args: 2 mouse clicks, 1 fill, 1 select_option

    assert args[0].function_name == "mouse_click"
    assert args[0].name == "xy"
    assert args[0].value == (34.0, 59.0)
    assert test_code[args[0].start_index : args[0].stop_index] == "34, 59"

    assert args[2].name == "from_xy"
    assert args[3].name == "to_xy"
    assert test_code[args[3].start_index : args[3].stop_index] == "to_x=100, to_y=200"


def manual_eval():
    """Manual test function that displays the resulting image."""
    import matplotlib.pyplot as plt

    # Create a white test image
    img = Image.new("RGB", (400, 300), "white")

    # Test action string with multiple function calls
    action_string = """mouse_click(100, 150)
fill("search_box", "hello world")
click("submit_btn")"""

    # Mock properties mapping bids to bounding boxes
    properties = {
        "search_box": {"bbox": (50, 50, 100, 50)},
        "submit_btn": {"bbox": (150, 100, 120, 30)},
    }

    # Annotate the image and get colored HTML
    html_result = overlay_utils.annotate_action(img, action_string, properties, colormap="tab10")

    # Display result
    plt.figure(figsize=(10, 6))
    plt.imshow(img)
    plt.axis("off")
    plt.show()

    print("HTML with colored arguments:")
    print(html_result)
    print("\nManual test completed!")


if __name__ == "__main__":
    manual_eval()
