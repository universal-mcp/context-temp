
from universal_mcp.servers import SingleMCPServer

from universal_mcp_context_temp.app import ContextApp

app_instance = ContextApp()

mcp = SingleMCPServer(
    app_instance=app_instance
)

if __name__ == "__main__":
    print(f"Starting {mcp.name}...")
    mcp.run(transport="streamable-http")
    