import asyncio
import json
import logging
import os
from contextlib import AsyncExitStack
from datetime import timedelta
from typing import Any

from mcp import ClientSession, StdioServerParameters, stdio_client
from mcp import Tool as MCPTool
from mcp.types import CallToolResult, ImageContent, TextContent

from agentlab.backends.browser.base import BrowserBackend, FunctionSpec, ToolCallAction, ToolSpec

logger = logging.getLogger(__name__)


class MCPClient:
    def __init__(self, config_path: str, read_timeout_seconds: int = 10) -> None:
        self.servers = self.load_config(config_path)
        self.sessions: dict[str, ClientSession] = {}
        self.tools: dict[str, MCPTool] = {}
        self.tool_to_server: dict[str, str] = {}
        self.read_timeout_seconds = read_timeout_seconds
        self.exit_stack = AsyncExitStack()
        self.loop = None

    def initialize(self):
        try:
            self.loop = asyncio.get_event_loop()
        except RuntimeError:
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)
        self.loop.run_until_complete(self.start_servers())

    async def ainitialize(self) -> None:
        await self.start_servers()

    async def start_servers(self):
        for server_name, server_params in self.servers.items():
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            stdio, write = stdio_transport
            session = await self.exit_stack.enter_async_context(
                ClientSession(
                    stdio, write, read_timeout_seconds=timedelta(seconds=self.read_timeout_seconds)
                )
            )
            await session.initialize()
            self.sessions[server_name] = session
            response = await session.list_tools()
            for tool in response.tools:
                if tool.name in self.tools:
                    raise Exception(
                        f"Tools conflict! Tool {tool.name} already provided by server '{self.tool_to_server[tool.name]}'"
                    )
                self.tools[tool.name] = tool
                self.tool_to_server[tool.name] = server_name
            logger.info(
                f"Connected to MCP server '{server_name}' with tools: {[tool.name for tool in response.tools]}"
            )
        logger.info(f"Started {len(self.servers)} MCP servers")

    def load_config(self, config_path) -> dict[str, StdioServerParameters]:
        assert os.path.exists(config_path), f"Config path {config_path} does not exist"
        self.config_path = config_path

        try:
            with open(config_path, "r") as f:
                self.config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse {config_path}, invalid json: {e}")
        try:
            server_configs: dict[str, dict] = self.config["mcpServers"]
            assert isinstance(server_configs, dict), "mcpServers must be a dict"
            assert len(server_configs) > 0, "mcpServers dict is empty"
        except Exception as e:
            raise ValueError(f"Failed to get MCP server configs from {config_path}: {e}")

        servers: dict[str, StdioServerParameters] = {}
        for server_name, server_config_dict in server_configs.items():
            try:
                server_config_dict = self.prepare_env_vars(server_config_dict)
                server_params = StdioServerParameters.model_validate(server_config_dict)
            except Exception as e:
                raise ValueError(f"Failed to parse server config {server_config_dict}: {e}")
            servers[server_name] = server_params
        logger.info(f"Loaded {len(servers)} MCP server configs from {config_path}")
        return servers

    def prepare_env_vars(self, server_config_dict: dict) -> dict:
        if server_env := server_config_dict.get("env"):
            for env_var, env_value in server_env.items():
                if (
                    env_var in os.environ and not env_value
                ):  # reuse existing env var value if not set in config
                    logger.info(f"Set mcp server env var {env_var} from current environment")
                    server_config_dict["env"][env_var] = os.environ[env_var]
        return server_config_dict

    def call_tool(self, tool_name: str, tool_args: dict[str, Any]) -> CallToolResult:
        result = self.loop.run_until_complete(self.acall_tool(tool_name, tool_args))
        return result

    async def acall_tool(self, tool_name: str, tool_args: dict[str, Any]) -> CallToolResult:
        server_name = self.check_tool_exists(tool_name)
        result = await self._call_tool(server_name, tool_name, tool_args)
        return result

    async def _call_tool(
        self, server_name: str, tool_name: str, tool_args: dict[str, Any]
    ) -> CallToolResult:
        try:
            session = self.sessions[server_name]
            result = await session.call_tool(tool_name, tool_args)
        except Exception as e:
            logger.exception(f"Error calling tool {tool_name}: {e}")
            raise e
        return result

    def check_tool_exists(self, tool_name):
        try:
            server_name = self.tool_to_server[tool_name]
        except KeyError:
            raise Exception(f"Tool {tool_name} not found in any of the MCP servers")
        return server_name

    def actions(self) -> tuple[ToolSpec]:
        return (
            ToolSpec(
                function=FunctionSpec(
                    name=tool.name, description=tool.description or "", parameters=tool.inputSchema
                )
            )
            for tool in self.tools.values()
        )

    async def close(self) -> None:
        await self.exit_stack.aclose()


class MCPBrowserBackend(BrowserBackend):
    config_path: str
    _mcp = None

    def initialize(self) -> None:
        self._mcp = MCPClient(config_path=self.config_path)
        self._mcp.initialize()

    def step(self, action: ToolCallAction) -> dict:
        contents = self.call_tool(action.function.name, action.function.arguments)
        text = "\n".join([c.text for c in contents if c.type == "text"])
        images = [c for c in contents if c.type == "image"]
        return {
            "pruned_html": text,
            "axtree_txt": text,
            "screenshot": images[-1] if images else None,
        }

    def call_tool(self, tool_name: str, arguments: dict) -> list[TextContent | ImageContent]:
        tool_result = self._mcp.call_tool(tool_name, arguments)
        if tool_result.isError:
            return [TextContent(text=f"Error calling tool {tool_name}: {tool_result.error}")]
        return tool_result.content

    def actions(self) -> tuple[ToolSpec]:
        return list(self._mcp.actions())

    def close(self) -> None:
        self._mcp.close()
