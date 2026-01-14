import os

from camel.toolkits.mcp_toolkit import MCPClient
from camel.utils.mcp_client import ServerConfig


def create_mcp_client(
    server_url: str = None,
    agent_id: str = "search_agent",
    agent_description: str = "Not described!",
    timeout: float = 300.0
) -> MCPClient:
    """Create and return an MCP client instance."""
    if agent_description == "Not described!":
        print("Warning: Agent description is not provided. Using default 'Not described!'.")

    base_url = f"http://localhost:5555/devmode/exampleApplication/privkey/{os.getenv("GAIA_TASK_ID")}/sse"
    
    import urllib.parse 
    params = {
    # "waitForAgents": 1,
    "agentId": agent_id,
    "agentDescription": agent_description
    }
    query_string = urllib.parse.urlencode(params)
    MCP_SERVER_URL = f"{base_url}?{query_string}"


    print(f"Connecting to MCP server at {base_url} with description '{agent_description}'")
    return MCPClient(ServerConfig(url = MCP_SERVER_URL, timeout=timeout, sse_read_timeout=timeout, terminate_on_close=True, prefer_sse=True), timeout=timeout)
