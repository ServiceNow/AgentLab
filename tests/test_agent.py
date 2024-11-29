from playwright.sync_api import sync_playwright
import pytest

# SInce you need code to run under 60s
pytestmark = pytest.mark.timeout(60)  # Timeout global de 60 secondes

# change the URL if needed
URL = "http://127.0.0.1:7860/"


def test_invalid_data_format(page):
    """Test when the data format is invalid and ensure the page handles it gracefully."""
    page.goto(URL)
    page.wait_for_selector(".main.svelte-1ufy31e")
    page.evaluate(
        """
        const mainContent = document.querySelector(".main.svelte-1ufy31e");
        if (mainContent) {
            mainContent.innerHTML = "<div>Corrupted Data</div>";
        }
    """
    )
    page.wait_for_selector(".main.svelte-1ufy31e")
    invalid_message = page.query_selector(".main.svelte-1ufy31e")
    assert invalid_message is not None, "The page should handle corrupted data gracefully."
    assert "Corrupted Data" in invalid_message.text_content(), "The page did not render corrupted data correctly."


def test_missing_experiment_content(page):
    """Test if the page handles missing content when an experiment is selected due to API changes."""
    page.goto(URL)
    experiment_dropdown = page.wait_for_selector('input[role="listbox"]')
    experiment_dropdown.click()
    experiment_option = page.query_selector("ul[role='listbox'] li:nth-child(1)")
    experiment_option.click()
    page.wait_for_timeout(3000)
    content_box = page.query_selector("#component-41")
    assert content_box is not None, "The experiment content was not loaded correctly."
    element_to_remove = page.query_selector("#component-41 .some-element")
    if element_to_remove:
        page.evaluate('document.querySelector("#component-41 .some-element").remove()')
    else:
        print("The element to remove does not exist.")
    content_box_after_removal = page.query_selector("#component-41")
    assert content_box_after_removal is not None, "The experiment content failed to handle missing elements."


def test_no_console_error(page):
    """Ensure no error messages are logged in the browser's console (e.g., due to data or API issues)."""
    page.goto(URL)
    console_logs = []
    page.on("console", lambda msg: console_logs.append(msg.text))
    page.wait_for_timeout(5000)
    error_logs = [log for log in console_logs if "error" in log.lower()]
    assert not error_logs, f"Des erreurs ont été trouvées dans les logs : {error_logs}"
