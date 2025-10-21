# Installing AgentLab and using the assistant

### Install uv
Simply run this: 
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```
alternatively, follow these [instructions](https://docs.astral.sh/uv/getting-started/installation/).

### Install Dependencies and Setup Environment
Once uv is installed, run the following command:

```bash
git clone https://github.com/ServiceNow/AgentLab.git
cd AgentLab
uv run playwright install
```
> **Note**: Sometimes, playwright may prompt you to install additional dependencies using `playwright install-deps`.

This command will:
- Automatically detect and install all required dependencies (including bgym and agentlab)
- Set up the Python environment
- Install Playwright and its browser dependencies

### Activate the Python Environment
Activate the environment created by uv:

```bash
source .venv/bin/activate
```

### Add OpenAI API Key
Add your key to your `.bashrc` or `.zshrc`.
```bash
echo 'export OPENAI_API_KEY="<REPLACE_THIS_BY_YOUR_KEY>"' >> ~/.bashrc
source ~/.bashrc
```
You can use other providers like Anthropic, OpenRouter, self hosted via vLLM. But you need to modify the config in the code.

### Launch assistant
```bash
agentlab-assistant
```
NOTE: our agents are not designed for user experience, but for benchmark performance. 

### Modify starting url and play with the code
Note that CAPTCHA to prevent agents are becoming more frequent, try a different starting url.

```bash
agentlab-assistant --start_url=https://duckduckgo.com/
```

Or modify the assistant script: [`src/agentlab/ui_assistant.py`](../../src/agentlab/ui_assistant.py)


