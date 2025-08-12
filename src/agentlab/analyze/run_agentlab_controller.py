from streamlit.web import cli

from pathlib import Path

CURR_DIR = Path(__file__).parent
agent_controller_path = CURR_DIR / "agent_controller.py"


def main():
    cli.main_run([str(agent_controller_path), "--server.port", "8501"])


if __name__ == "__main__":
    main()
