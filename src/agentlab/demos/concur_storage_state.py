from playwright.sync_api import sync_playwright
from agentlab.experiments.exp_utils import RESULTS_DIR


def manual_login_and_save_state():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)  # Set headless=False to see the browser
        context = browser.new_context()
        page = context.new_page()

        # Navigate to the login page
        page.goto("https://servicenow.okta.com/app/UserHome?session_hint=AUTHENTICATED")

        # Wait for the user to complete the login manually
        input(
            "Please log in manually in the opened browser window. Press Enter here once you're done..."
        )

        # Save the state after manual login
        context.storage_state(path=RESULTS_DIR / "concur_state.json")

        print("Authentication state has been saved to state.json.")

        # Cleanup
        page.close()
        context.close()
        browser.close()


manual_login_and_save_state()
