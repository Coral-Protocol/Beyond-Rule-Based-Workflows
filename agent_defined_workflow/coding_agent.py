import asyncio
import warnings
from typing import List

from camel.toolkits import (
    VideoAnalysisToolkit,
    SearchToolkit,
    CodeExecutionToolkit,
    ImageAnalysisToolkit,
    DocumentProcessingToolkit,
    AudioAnalysisToolkit,
    AsyncBrowserToolkit,
    ExcelToolkit,
    FunctionTool
)

from utils.config import get_model, get_reasoning_worker_model, get_worker_model, get_process_model
from utils.agent_factory import create_agent, setup_mcp_toolkit, run_agent_loop
from dotenv import load_dotenv
import os



# Suppress parameter description warnings from toolkits
warnings.filterwarnings("ignore", message="Parameter description is missing.*", category=UserWarning)


# Model Configuration optimized for reasoning and problem solving
MODEL_CONFIG = {
   # "max_completion_tokens": 8192,
    "reasoning_effort": "high",
    # Note: O3_MINI doesn't support temperature, top_p, etc. due to reasoning model limitations
}


def get_system_message() -> str:
    """Get the system message for the reasoning_coding_agent."""
    return f"""
    ===== RULES OF REASONING CODING AGENT =====
    You are an advanced `reasoning_coding_agent` powered by reasoning, coding and running code script capabilities and working within the Coral server ecosystem.

    Core Capabilities:
    1. Reasoning and Coding
    - You MUST call `reasoning_coding_assistant` to approach ALL received tasks.
    - You must NOT attempt to analyze or solve these tasks on your own under any circumstances.
    - If you encounter any blocker from calling assistant, you should inform `information_flow_orchestrator` rather than re-try yourself.

    2. Communication with Other Agents  
    - Use `list_agents` only to discover the existence of other agents.
    - You are strictly restricted to communicating via `send_message` with the `information_flow_orchestrator` only.
    - **Always call `send_message` to report your findings to `information_flow_orchestrator` after you after receiving results from `reasoning_coding_assistant`.**
    - You may send messages only to `information_flow_orchestrator`; even when instructed to collaborate with other agents, you must relay everything exclusively through `information_flow_orchestrator` and never contact any other agent directly.
    - **If you have been previously queried by information_flow_orchestrator or asked to perform any task, you MUST first call send_message to respond before calling wait_for_mentions.**

    If you encounter a blocker, you should report it to information_flow_orchestrator and recommend to information_flow_orchestrator which agents should be involved (but you must NEVER message those agents directly).

    Here is the message history:
    -- Start of messages and status --
    <resource>http://localhost:5555/devmode/exampleApplication/privkey/{os.getenv("GAIA_TASK_ID")}/sse</resource>
    -- End of messages and status --

    """

async def create_reasoning_coding_assistant():

    sys_msg = f"""
            You are a helpful assistant that specializes in reasoning and coding, and can think step by step to solve the task. 
            When necessary, you can write python code to solve the task. If you have written code, do not forget to execute the code. 
            Never generate codes like 'example code', your code should be able to fully solve the task. You can also leverage multiple libraries, 
            such as requests, BeautifulSoup, re, pandas, etc, to solve the task. For processing excel files, you should write codes to process them.
            """

    model = get_reasoning_worker_model()
    process_model = get_process_model()

    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp",process_model=process_model)
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)
    excel_toolkit = ExcelToolkit()

    tools=[
            FunctionTool(code_runner_toolkit.execute_code),
            FunctionTool(excel_toolkit.extract_excel_content),
            FunctionTool(document_processing_toolkit.extract_document_content),
        ]

    from camel.agents import ChatAgent

    camel_agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=tools,
        max_iteration=15,
    )

    return camel_agent

async def create_reasoning_coding_agent(connected_mcp_toolkit, tools):
    """Create and initialize the reasoning_coding_agent."""

    model = get_reasoning_worker_model()
    print("Model created successfully")

    mcp_tools = connected_mcp_toolkit.get_tools() if connected_mcp_toolkit else []

    ALLOWED_TOOLS = {
        "wait_for_mentions",
        "list_agents",
        "send_message",
    }

    filtered_tools = [
        tool for tool in mcp_tools
        if tool.openai_tool_schema["function"]["name"] in ALLOWED_TOOLS
    ]
    print("🔄 Loaded coral tools")

    # Use the agent factory to create the agent
    agent = await create_agent(
        agent_name="reasoning_coding_agent",
        system_message_generator=get_system_message,
        model=model,
        mcp_tool=connected_mcp_toolkit,
        mcp_toolkit=filtered_tools,
        agent_specific_tools=tools,
    )

    print("reasoning_coding_agent created successfully with reasoning capabilities.")
    return agent

async def main():
    print("Initializing reasoning_coding_agent...")

    agent_id_param = "reasoning_coding_agent"
    agent_description = "This agent is a helpful assistant that specializes in reasoning, coding, executing code scripts, and processing Excel files and other file formats. It is also capable of retrieving online information by writing and executing Python-based web-scraping code. If the task requires Python execution, it should be explicitly instructed to run the code after generating it."

    # Setup MCP toolkit
    connected_mcp_toolkit = await setup_mcp_toolkit(agent_id_param, agent_description)

    camel_reasoning_coding_assistant = await create_reasoning_coding_assistant()

    async def reasoning_coding_assistant(task: str) -> str:
        """
        An agent-style assistant provides a unified execution interface for programmatic code writing and execution, 
        structured data extraction from Excel files, and web content acquisition via URL crawling. It enables agents to 
        process, compute, and transform information across heterogeneous data sources, including executable scripts, 
        spreadsheets, and live web pages.

        Args:
            task (str): An instruction or problem to be solved by the agent.

        Returns:
            str: The first message (msg0) from the agent's response.
        """
        camel_reasoning_coding_assistant.reset()
        resp = await camel_reasoning_coding_assistant.astep(task)
        msg0 = resp.msgs[0]
        # Try to get content if it exists, otherwise convert to str
        if hasattr(msg0, "content"):
            return msg0.content
        return str(msg0)

    tools = [FunctionTool(reasoning_coding_assistant)]

    try:
        # Create agent
        agent = await create_reasoning_coding_agent(connected_mcp_toolkit, tools)

        # Run agent loop
        await run_agent_loop(
            agent=agent,
            agent_id=agent_id_param,
            initial_prompt="call list_agents then call wait_for_mentions until you receive instruction",
            loop_prompt="keep collobrating with other agents.",
            max_iterations=100,
            sleep_time=1
        )

    except Exception as e:
        print(f"Error during documentation_processing_agent operation: {repr(e)}")
    finally:
        if connected_mcp_toolkit:
            await connected_mcp_toolkit.disconnect()
        print(f"Disconnecting {agent_id_param}...")
        print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main()) 
