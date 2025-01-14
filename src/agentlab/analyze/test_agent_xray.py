import pytest
from playwright.sync_api import sync_playwright

# Base URL for the application (CHANGE IF NECESSARY)
BASE_URL = "http://127.0.0.1:7861"  # Ensure the application is running locally

@pytest.fixture(scope="module")
def browser():
    """Set up Playwright browser for the test session."""
    with sync_playwright() as p:
        # Launch a headless Chromium browser
        browser = p.chromium.launch(headless=True)
        yield browser  # Provide the browser instance to the tests
        browser.close()  # Close the browser after tests are complete

@pytest.fixture
def page(browser):
    """Provide a new browser page for each test."""
    # Create a new browser context to isolate tests
    context = browser.new_context()
    # Open a new page in the browser
    page = context.new_page()
    # Navigate to the base URL of the application
    page.goto(BASE_URL)
    yield page  # Provide the page instance to the test
    context.close()  # Close the browser context after the test

def test_data_container_load(page):
    """Verify the data container loads with content."""
    # Wait for the data container to appear on the page
    container = page.wait_for_selector("#data-container", timeout=10000)
    # Assert that the data container is visible
    assert container.is_visible(), "[ERROR] Data container not visible."

    # Locate all data items within the container
    data_items = page.locator("#data-container .data-item")
    # Assert that there is at least one data item present
    assert data_items.count() > 0, "[ERROR] No data items found."

def test_dropdown_selection(page):
    """Test the experiment directory dropdown functionality."""
    # Locate the dropdown input field by its aria-label
    dropdown_input = page.locator("input[aria-label='Experiment Directory']")
    # Wait for the dropdown input field to become visible
    dropdown_input.wait_for(state="visible", timeout=10000)
    # Click the dropdown to expand options
    dropdown_input.click()

    # Locate the dropdown options container
    dropdown_options = page.locator("ul[role='listbox']")
    # Wait for the dropdown options to appear
    dropdown_options.wait_for(state="visible", timeout=10000)

    # Fetch all the options available in the dropdown
    all_options = dropdown_options.locator("li").all_text_contents()

    # Assert that the dropdown contains at least one option
    assert all_options, "[ERROR] No dropdown options found."

    # Select the first option from the dropdown
    option_text = all_options[1]
    dropdown_options.locator({option_text}).click()

    # Verify the selected value is reflected in the input field
    assert dropdown_input.input_value() == option_text, "[ERROR] Selected value mismatch."

def test_navigation_between_tabs(page):
    """Verify navigation between tabs."""
    # Locate the tab element by its role and aria-controls attribute
    tab = page.locator("button[role='tab'][aria-controls='component-11']")
    # Wait for the tab to become visible
    tab.wait_for(state="visible", timeout=10000)
    # Click the tab to activate it
    tab.click()

    # Assert that the tab is now active
    assert tab.get_attribute("aria-selected") == "true", "[ERROR] Tab not activated."

