import os
import asyncio
from mcp.client.stdio import stdio_client, StdioServerParameters
from mcp import ClientSession
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType

def create_mcp_tool(session: ClientSession, tool_name: str, description: str = None):
    """Create a LangChain tool from an MCP tool"""
    
    async def async_mcp_tool(query: str) -> str:
        try:
            print(f"🔧 Calling MCP tool '{tool_name}' with query: {query}")
            response = await session.call_tool(tool_name, {"query": query, "k": 3})
            print(f"✅ Tool response received: {str(response)[:200]}...")
            return str(response)
        except Exception as e:
            error_msg = f"Error calling tool {tool_name}: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg
    
    def sync_mcp_tool(query: str) -> str:
        return asyncio.run(async_mcp_tool(query))
    
    desc = description or f"Search for similar documents using {tool_name}"
    
    return Tool(
        name=tool_name,
        description=desc,
        func=sync_mcp_tool,
        coroutine=async_mcp_tool
    )

async def run_agent_with_custom_mcpclient():
    server_params = StdioServerParameters(
        command="python",
        args=["mcp_server.py"],
        env=os.environ,
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the session
            await session.initialize()
            
            # List available tools
            list_result = await session.list_tools()
            available_tools = list_result.tools
            print(f"🛠️ Available tools: {[tool.name for tool in available_tools]}")
            
            # Create LangChain tools - only use query_similar_documents for now
            tools = []
            for tool_info in available_tools:
                if tool_info.name == "query_similar_documents":  # Focus on the search tool
                    tool = create_mcp_tool(
                        session, 
                        tool_info.name,
                        getattr(tool_info, 'description', "Search for similar documents based on vector similarity")
                    )
                    tools.append(tool)
                    print(f"✅ Created tool: {tool_info.name}")
            
            if not tools:
                print("❌ No suitable tools found")
                return
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash",
                temperature=0,
                google_api_key=os.getenv("GEMINI_API_KEY"),
            )
            
            agent = initialize_agent(
                tools,
                llm,
                agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                handle_parsing_errors=True,
                max_iterations=3,
                early_stopping_method="generate"
            )

            user_query = "Where is Pune?"
            print(f"\n🤖 Running agent with query: {user_query}")
            
            try:
                # Use ainvoke instead of the deprecated arun
                response = await agent.ainvoke({"input": user_query})
                print("\n" + "="*60)
                print("✅ Agent response:")
                print("="*60)
                print(response.get('output', response))
            except Exception as e:
                print(f"❌ Agent execution error: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_agent_with_custom_mcpclient())