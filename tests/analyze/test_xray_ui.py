import subprocess
import pytest
import time
from pathlib import Path
import re
from agentlab.analyze.agent_xray import get_directory_contents
from agentlab.experiments.exp_utils import RESULTS_DIR
from playwright.sync_api import sync_playwright, Page, Locator


@pytest.fixture(scope="session")
def start_agentlab_xray_server():
    """
    The fixture does the following:
    - Starts the agentlab-xray server in a subprocess.
    - Waits for the server to start and captures the Gradio URL from the server's output.
    - Yields the Gradio URL and the server process for use in tests.
    - Terminates the server process after the test session ends.
    """
    # Start the agentlab-xray server in a subprocess
    server_process = subprocess.Popen(
        ["agentlab-xray"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    time.sleep(5)
    # Read the server output and search for the Gradio URL in the logs
    gradio_url = None
    for line in server_process.stdout:
        if "Running on" in line:  # Look for lines that show the Gradio URL
            match = re.search(r"(http://\S+)", line)
            if match:
                gradio_url = match.group(1)
                break

    if not gradio_url:
        gradio_url = "http://127.0.0.1:7860"  # Default to the known URL if not found

    yield gradio_url, server_process

    server_process.terminate()
    server_process.wait()


@pytest.fixture(scope="function")
def browser_page(start_agentlab_xray_server):
    """
    Sets up a Playwright browser, context, and page for testing.

    This fixture launches a Chromium browser, creates a new browser context,
    and opens a new page that navigates to the provided Gradio app URL.
    After the test completes, the context and browser are properly closed.

    Args:
        start_agentlab_xray_server: The fixture that starts the server.

    Yields:
        Page: A Playwright page object for interacting with the application.
    """
    gradio_url, server_process = start_agentlab_xray_server

    with sync_playwright() as p:
        browser = p.chromium.launch(slow_mo=500)  # Takes care of UI element delays
        context = browser.new_context()
        page = context.new_page()
        page.goto(gradio_url)
        yield page
        context.close()
        browser.close()


def jump_to_exp(page: Page, exp: str):
    """
    Navigates to the specified experiment in the application.

    Args:
        page (Page): The Playwright page object.
        exp (str): The name of the experiment to navigate to.

    Returns:
        Page: The updated Playwright page object after navigation.
    """
    page.get_by_label("Experiment Directory").click()
    page.wait_for_load_state()
    page.get_by_text(f"{exp}").click()
    page.wait_for_load_state()
    return page


def save_error_screenshot(page: Page, error_locator: Locator, name: str):
    """
    Saves a screenshot of the page using PLaywright locator object.

    Args:
        page (Page): The Playwright page object.
        error_locator (Locator): The locator for the error element to focus on.
        name (str): The filename for the screenshot.
    """
    try:
        screenshot_dir = Path("xray_ui_error_screenshots")
        screenshot_dir.mkdir(parents=True, exist_ok=True)
        error_locator.focus()
        error_locator.scroll_into_view_if_needed()
        page.wait_for_load_state()
        page.screenshot(path=screenshot_dir.joinpath(f"{name}.png"), full_page=True)
    except Exception as e:
        print(f"Failed to save screenshot for {name}: {e}")


def check_visible_errors(page: Page, save_screenshot: bool = False, screenshot_name: str = None):
    """
    Checks for visible error messages on the page and optionally saves a screenshot.

    Args:
        page (Page): The Playwright page object to inspect.
        save_screenshot (bool): Whether to save a screenshot of the page if an error is found. Default is False.
        screenshot_name (str): The filename for the screenshot, if saved. Default is None.

    Returns:
        bool: True if any visible error is found, False otherwise.
    """
    error_found = False
    err_locators = page.get_by_text(re.compile("^Error$")).all()
    for err_locator in err_locators:
        if err_locator.is_visible():
            if save_screenshot:
                save_error_screenshot(page, err_locator, screenshot_name)
            error_found = True
    return error_found


def test_agent_tab(browser_page):
    """
    Tests the "Agent" tab in all experiments for visible errors.

    This function navigates through the experiments listed in the results directory,
    opens the "Agent" tab for each experiment, and checks for any visible error messages.
    If an error is found, a screenshot is saved for debugging.

    Args:
        page (Page): The Playwright page object used for navigation and interaction.

    Asserts:
        That no visible errors are found across all experiments.
    """
    page = browser_page
    experiments = get_directory_contents(RESULTS_DIR)  # choices from the dropdown
    error_found = False
    for i, exp in enumerate(experiments):
        page = jump_to_exp(page, exp)
        page.get_by_role("tab", name="Select Agent").click()
        page.wait_for_load_state()
        err = check_visible_errors(page, save_screenshot=True, screenshot_name=f"exp_{i}_agent")
        if err:
            error_found = True
    assert not error_found


# Test case to interact with the dropdown and select experiments
def test_task_and_seed_buttons_of_exp(browser_page):
    """
    Tests for visible errors when selecting tasks and seeds in all experiments.

    This function iterates through each experiment, selects the "Select Task and Seed" tab,
    and interacts with all task/seed buttons. If an error is found during any interaction,
    a screenshot is saved. The function asserts that no errors are found across all experiments.

    Args:
        page (Page): The Playwright page object used for navigation and interaction.

    Asserts:
        That no visible errors are found when selecting tasks and seeds in all experiments.
    """
    page = browser_page
    experiments = get_directory_contents(RESULTS_DIR)  # Exp list from dropdown
    error_found = False
    for i, exp in enumerate(experiments):
        page = jump_to_exp(page, exp)
        page.get_by_role("tab", name="Select Agent").click()
        page.wait_for_load_state()
        page.get_by_role("tab", name="Select Task and Seed").click()
        page.wait_for_load_state()
        buttons = page.get_by_role("button").filter(has_text="seed").all()
        for j, button in enumerate(buttons):
            button.click(force=True)
            button.highlight()
            page.wait_for_load_state()
            err = check_visible_errors(
                page, save_screenshot=True, screenshot_name=f"exp_{i}_button_{j}"
            )
            if err:
                error_found = True
    assert not error_found


def test_all_tabs_of_exp(browser_page):
    """
    Tests all tabs of all experiments for visible errors.

    This function navigates through each experiment and all of its tabs, checking for
    any visible errors. If an error is found, a screenshot is saved. The function
    asserts that no errors are detected across all experiments and tabs.

    Args:
        page (Page): The Playwright page object used for navigating and interacting.

    Asserts:
        That no visible errors are found across all experiments and tabs.
    """
    page = browser_page
    experiments = get_directory_contents(RESULTS_DIR)  # Get the list of choices from the dropdown
    error_found = False
    for i, exp in enumerate(experiments):
        page = jump_to_exp(page, exp)
        tabs = page.get_by_role("tab").all()
        for j, tab in enumerate(tabs):
            tab.click()
            page.wait_for_load_state()
            err = check_visible_errors(
                page, save_screenshot=True, screenshot_name=f"exp_{i}_tab_{j}"
            )
            if err:
                error_found = True
    assert not error_found


if __name__ == "__main__":
    # for debugging
    server_process = subprocess.Popen(
        ["agentlab-xray"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    time.sleep(5)
    test_functions = [test_agent_tab, test_task_and_seed_buttons_of_exp, test_all_tabs_of_exp]
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False, slow_mo=400)  # see the test in live action
        context = browser.new_context()
        page = context.new_page()
        page.goto("http://127.0.0.1:7860/")

        for test_func in test_functions:
            try:
                test_func(page)
            except Exception as e:
                print(f"Error occurred in {test_func.__name__}:")
                print(f"{str(e)}")
                continue

        context.close()
        browser.close()
        server_process.terminate()
        server_process.wait()
