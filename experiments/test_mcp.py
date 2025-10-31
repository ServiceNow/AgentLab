from tapeagents.environment import FunctionCall
from tapeagents.mcp import ToolCallAction

from agentlab.backends.browser.mcp_playwright import MCPPlaywright
from agentlab.benchmarks.miniwob.task import get_miniwob_tasks


def main():
    tasks = get_miniwob_tasks()
    task = tasks[0]
    setup_js = task.get_setup_js()

    backend = MCPPlaywright()
    print("="*100)
    # 1. goto task url
    print("URL: ", task.url)
    obs = backend.call_tool("browser_navigate", {"url": task.url})
    print("------")
    print(obs)
    print("-"*100)

    # 2. eval js
    obs = backend.run_js(setup_js)
    print("------")
    print(obs)
    print("-"*100)

    # 3. validate
    print("\n\nVALIDATE")
    js = task.get_task_validate_js()
    print(js)
    obs = backend.run_js(js)
    print("------")
    print(obs)
    print("-"*100)

if __name__ == "__main__":
    main()



    