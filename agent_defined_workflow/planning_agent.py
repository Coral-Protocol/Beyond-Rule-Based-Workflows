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

from utils.config import get_model, get_worker_model, get_image_model, get_audio_model
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
    """Get the system message for the planner_agent."""
    return f"""
    ===== RULES OF PLANNER AGENT =====
    You are an advanced `planner_agent` to decompose task into subtask, replan the task based on previosu attempted trajactories and cooperate with other agents in coral server.

    Core Responsibilities:

    1. Task Decompostion 
    - You must send all decomposed subtasks to information_flow_orchestrator in the format of a numbered list within <tasks> tags, as shown below:
        <tasks>
        <task>Subtask 1</task>
        <task>Subtask 2</task>
        </tasks>
    - You MUST NOT explicitly mention what agents and what tools to use in the subtasks, just let the agent decide what to do.
    - Though it's not a must, you should try your best effort to make each subtask achievable for an agent.

    2. Task Progress Reasoning
    - When asked to perform tasks including but not limited to verification, critique, assessing the reliability of intermediate results, replanning, reflection, questioning, or critique, 
        provide the necessary reasoning to support task progress.
    
    3. Communication with Other Agents  
    - Use `list_agents` only to discover the existence of other agents.
    - You are strictly restricted to communicating via `send_message` with the `information_flow_orchestrator` only.
    - You may send messages only to `information_flow_orchestrator`; even when instructed to collaborate with other agents, you must relay everything exclusively through `information_flow_orchestrator` and never contact any other agent directly.
    - If you have been previously queried by information_flow_orchestrator or asked to perform any task, you MUST first call send_message to respond before calling wait_for_mentions.
    
    If you simply need more information, don't overthink it, just ask `information_flow_orchestrator` for help then try again when it respond.

    Here is the message history:
    -- Start of messages and status --
    <resource>http://localhost:5555/devmode/exampleApplication/privkey/{os.getenv("GAIA_TASK_ID")}/sse</resource>
    -- End of messages and status --

    """

async def create_planner_agent(connected_mcp_toolkit, tools):
    """Create and initialize the planner_agent."""

    model = get_model()
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
        agent_name="planner_agent",
        system_message_generator=get_system_message,
        model=model,
        mcp_tool=connected_mcp_toolkit,
        mcp_toolkit=filtered_tools,
        agent_specific_tools=tools,
    )

    print("planner_agent created successfully with reasoning capabilities.")
    return agent

async def main():
    print("Initializing planner_agent...")

    agent_id_param = "planner_agent"
    agent_description = "This agent is a helpful assistant that can decompose task into subtask, adjust task insturctions and provide addidtional reasoning based on previous attempted trajactories. While this agent CANNOT execute task."

    # Setup MCP toolkit
    connected_mcp_toolkit = await setup_mcp_toolkit(agent_id_param, agent_description)

    tools = []

    try:
        # Create agent
        agent = await create_planner_agent(connected_mcp_toolkit, tools)

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
        print(f"Error during planner_agent operation: {repr(e)}")
    finally:
        if connected_mcp_toolkit:
            await connected_mcp_toolkit.disconnect()
        print(f"Disconnecting {agent_id_param}...")
        print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main()) 
