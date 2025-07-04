
from universal_mcp.servers import SingleMCPServer

from universal_mcp_context_temp.app import ContextTempApp

app_instance = ContextTempApp()

mcp = SingleMCPServer(
    app_instance=app_instance
)

if __name__ == "__main__":
    print(f"Starting {mcp.name}...")
    mcp.run()


