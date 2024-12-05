from typing import Callable
import os
import pytest
import subprocess
import time

from playwright.sync_api import sync_playwright, Page
from pathlib import Path


AGENTLAB_XRAY_URL = "http://127.0.0.1:7860"
PATH_URL = Path("tests/data/test_study")
TEST_EXPERIMENT_PATH = "tests/data/test_study"

os.environ["AGENTXRAY_APP_PORT"] = "7860"


def launch_gradio_app():
    # Launch the Gradio app in a separate process
    process = subprocess.Popen(
        ["python", "src/agentlab/analyze/agent_xray.py", "--experiment_path", TEST_EXPERIMENT_PATH],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    print(process)
    time.sleep(5)  # Wait for the app to start (adjust based on app startup time)
    return process


def stop_gradio_app(process: subprocess.Popen[bytes]):
    # Terminate the Gradio app process
    process.terminate()
    process.wait()


def run_playwright_with_test(test_function: Callable[[Page], None]):

    # Launch the Gradio app
    gradio_process = launch_gradio_app()

    try:
        # Use Playwright to test the app
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()

            # Navigate to the Gradio app
            page.goto(AGENTLAB_XRAY_URL)

            test_function(page)

            # Close the browser
            browser.close()

    finally:
        # Stop the Gradio app
        stop_gradio_app(gradio_process)


@pytest.mark.playwright
def test_clicking_dropdown():
    """Check that the experiment directory dropdown can be clicked"""

    def click_dropdown(page: Page):
        dropdown = page.locator("#experiment_directory_dropdown")
        dropdown.select_option(
            "2024-08-01_10-20-52_GenericAgent_on_miniwob.ascending-numbers_68_b6312d"
        )

    run_playwright_with_test(click_dropdown)


if __name__ == "__main__":
    test_clicking_dropdown()
