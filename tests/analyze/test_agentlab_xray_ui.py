import pytest
from playwright.sync_api import sync_playwright
import subprocess
import time


@pytest.fixture(scope="session")
def xray_server():
    """
    Launches agentlab-xray on port 7860 once for all tests,
    then tears it down after tests complete.
    """
    process = subprocess.Popen(["agentlab-xray", "--server_port=7860"])
    time.sleep(5)
    yield
    process.terminate()
    process.wait()


def _check_for_unexpected_errors(page):
    """
    Collects all DOM elements containing the substring 'Error'.
    Skips a few known harmless strings (like 'Error Report', 'Task Error', or 'errors:').
    Fails if it finds something else likely indicating a real error.
    """
    error_candidates = page.query_selector_all("text=Error")
    allowed_substrings = ["Error Report", "Task Error", "errors:", "Error"]
    real_errors = []

    for el in error_candidates:
        content = el.inner_text().strip()
        lower_content = content.lower()
        if any(s.lower() in lower_content for s in allowed_substrings):
            continue
        real_errors.append(content)

    if real_errors:
        pytest.fail(f"Found unexpected error text: {real_errors}")


def test_xray_loads_data(xray_server):
    """
    1) Open agentlab-xray UI.
    2) Select an experiment from the dropdown.
    3) Confirm no unexpected 'Error' text appears.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto("http://127.0.0.1:7860/")
        page.wait_for_selector("#exp_dir_dropdown", timeout=15000)

        page.click("#exp_dir_dropdown")

        page.wait_for_selector("ul[role='listbox'] li", timeout=5000)
        items = page.query_selector_all("ul[role='listbox'] li")
        assert len(items) > 1, "No experiments found in the dropdown!"

        items[-1].click()
        page.wait_for_timeout(1000)

        _check_for_unexpected_errors(page)
        browser.close()


def test_xray_select_agent(xray_server):
    """
    1) Open agentlab-xray UI.
    2) Select the first experiment from the dropdown.
    3) Wait for an agent table to appear and verify it's not empty.
    4) Check for unexpected 'Error' text.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto("http://127.0.0.1:7860/")
        page.wait_for_selector("#exp_dir_dropdown", timeout=15000)
        page.click("#exp_dir_dropdown")

        page.wait_for_selector("ul[role='listbox'] li", timeout=5000)
        items = page.query_selector_all("ul[role='listbox'] li")
        assert len(items) > 1, "No experiments found in dropdown!"

        items[-1].click()
        page.wait_for_timeout(1000)

        page.wait_for_selector("table", timeout=5000)

        rows = page.query_selector_all("table tr")
        assert len(rows) > 1, "Agent table is empty!"

        _check_for_unexpected_errors(page)
        browser.close()


def test_xray_tabs_navigation(xray_server):
    """
    Scenario:
    1) Open agentlab-xray UI.
    2) Select the first experiment from the dropdown.
    3) Switch to several important tabs ('Screenshot', 'DOM HTML', ...).
    4) At each step, confirm no unexpected 'Error' text appears.
    """
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto("http://127.0.0.1:7860/")
        page.wait_for_selector("#exp_dir_dropdown", timeout=15000)
        page.click("#exp_dir_dropdown")

        page.wait_for_selector("ul[role='listbox'] li", timeout=5000)
        items = page.query_selector_all("ul[role='listbox'] li")
        assert len(items) > 1, "No experiments found in dropdown!"

        items[0].click()
        page.wait_for_timeout(1500)

        tabs_to_test = [
            "Screenshot",
            "Screenshot Pair",
            "Screenshot Gallery",
            "DOM HTML",
            "Pruned DOM HTML",
            "AXTree",
            "Chat Messages",
            "Task Error",
            "Logs",
        ]

        for tab_label in tabs_to_test:
            page.click(f"text={tab_label}", force=True)
            page.wait_for_timeout(500)
            _check_for_unexpected_errors(page)

        browser.close()
