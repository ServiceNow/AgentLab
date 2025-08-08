# Welcome to the Tutorial: Creating Your First Web Agent!

In this tutorial, we'll guide you through creating your first web agent. We'll start by setting up our Python environment using **uv**, a fast Python package and project manager.

## Prerequisites

### 1. Install uv
Simply run this: 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
alternatively, follow these [instructions](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Install Dependencies and Setup Environment
Once uv is installed, run the following command:

```bash
uv run playwright install
```

This command will:
- Automatically detect and install all required dependencies (including bgym and agentlab)
- Set up the Python environment
- Install Playwright and its browser dependencies

Let's get started building your web agent!

### Launch assistant
```bash
agentlab-assistant
```