import pytest
from playwright.sync_api import sync_playwright


@pytest.fixture(scope="module")
def setup_playwright():
    """Set up and tear down Playwright."""
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        yield browser
        browser.close()


@pytest.fixture
def load_xray_page(setup_playwright):
    """Launch browser and load AgentLab-Xray."""
    browser = setup_playwright
    context = browser.new_context()
    page = context.new_page()
    print("Navigating to http://localhost:7860...")
    page.goto("http://localhost:7860")  # Ensure AgentLab-Xray is running locally
    print("Page loaded successfully.")
    yield page
    print("Closing browser context...")
    context.close()


def test_experiment_directory_selection(load_xray_page):
    """Test experiment directory selection functionality."""
    page = load_xray_page

    # Wait for the input field (Experiment Directory) to load
    print("Waiting for experiment directory input field...")
    dropdown_input = page.locator("input[aria-label='Experiment Directory']")
    dropdown_input.wait_for(state="visible", timeout=10000)

    # Click on the dropdown input to expand options
    print("Clicking the experiment directory input...")
    dropdown_input.click()

    # Wait for the dropdown options container to appear
    print("Waiting for dropdown options to appear...")
    dropdown_options = page.locator("#dropdown-options")
    dropdown_options.wait_for(state="visible", timeout=10000)

    # Fetch all options from the dropdown
    print("Fetching dropdown options...")
    options = dropdown_options.locator("div")  # Adjust if options have a different tag or class
    all_options = options.all_text_contents()

    # Debug: Log the available options
    print(f"Available options: {all_options}")
    assert len(all_options) > 0, "[ERROR] No options found in the dropdown."

    # Select a valid option from the dropdown
    option_text = all_options[0]  # Select the first option (update as per test requirements)
    print(f"Selecting option: {option_text}")
    target_option = dropdown_options.locator(f"text='{option_text}'")
    assert target_option.is_visible(), f"[ERROR] Option '{option_text}' not visible."
    target_option.click()

    # Verify the selected value in the input field
    selected_value = dropdown_input.input_value()
    print(f"Selected experiment directory: {selected_value}")
    assert selected_value == option_text, "[ERROR] Selected value mismatch."




def test_navigation_tabs(load_xray_page):
    """Test navigation between different tabs."""
    page = load_xray_page

    # Wait for and navigate to the "Select Agent" tab
    print("Navigating to 'Select Agent' tab...")
    select_agent_tab = page.locator("button[role='tab'][aria-controls='component-11']")
    select_agent_tab.wait_for(state="visible", timeout=10000)
    select_agent_tab.click()

    # Verify the tab is active
    assert select_agent_tab.get_attribute("aria-selected") == "true", "'Select Agent' tab not activated."
    print("Successfully navigated to 'Select Agent' tab.")


