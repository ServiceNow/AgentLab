# Installing AgentLab and using the assistant

### 1. Install uv
Simply run this: 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
alternatively, follow these [instructions](https://docs.astral.sh/uv/getting-started/installation/).

### 2. Install Dependencies and Setup Environment
Once uv is installed, run the following command:

```bash
git clone https://github.com/ServiceNow/AgentLab.git
cd AgentLab
uv run playwright install
```

This command will:
- Automatically detect and install all required dependencies (including bgym and agentlab)
- Set up the Python environment
- Install Playwright and its browser dependencies

### 3. Activate the Python Environment
Activate the environment created by uv:

```bash
source .venv/bin/activate
```

### Add OpenAI API Key
Add your key to your `.bashrc` or `.zshrc`.
```bash
export OPENAI_API_KEY="<content_of_the_key>"
```
You can use other providers like Anthropic, OpenRouter, self hosted via vLLM. But you need to modify the config in the code.

### 4. Launch assistant
```bash
agentlab-assistant
```