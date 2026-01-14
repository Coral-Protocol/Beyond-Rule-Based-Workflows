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

from utils.config import get_model, get_worker_model, get_image_model, get_audio_model, get_process_model
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
    """Get the system message for the documentation_processing_agent."""
    return f"""
    ===== RULES OF DOCUMENTATION PROCESSING AGENT =====
    You are an advanced `document_processing_agent` powered by documentation processing capabilities and working within the Coral server ecosystem, but you are not able to run code script.

    Core Capabilities:
    1. Process Documents and Multimodal Data
    - You MUST call `documentation_processing_assistant` to approach ALL received tasks.
    - You must NOT attempt to analyze or solve these tasks on your own under any circumstances.
    - If you encounter any blocker from calling assistant, you should inform `information_flow_orchestrator` rather than re-try yourself.

    2. Communication with Other Agents  
    - Use `list_agents` only to discover the existence of other agents.
    - You are strictly restricted to communicating via `send_message` with the `information_flow_orchestrator` only.
    - **Always call `send_message` to report your findings to `information_flow_orchestrator` after you after receiving results from `document_processing_assistant`.**
    - You may send messages only to `information_flow_orchestrator`; even when instructed to collaborate with other agents, you must relay everything exclusively through `information_flow_orchestrator` and never contact any other agent directly.
    - **If you have been previously queried by information_flow_orchestrator or asked to perform any task, you MUST first call send_message to respond before calling wait_for_mentions.**
    
    
    If you encounter a blocker, you should report it to information_flow_orchestrator and recommend to information_flow_orchestrator which agents should be involved (but you must NEVER message those agents directly).


    Here is the message history:
    -- Start of messages and status --
    <resource>http://localhost:5555/devmode/exampleApplication/privkey/{os.getenv("GAIA_TASK_ID")}/sse</resource>
    -- End of messages and status --

    """

async def create_documentation_processing_assistant():

    sys_msg = f"""
            You are a helpful assistant that can process documents and multimodal data, such as images, audio, and video.
            """

    model = get_worker_model()
    image_analysis_model = get_image_model()
    audio_reasoning_model = get_audio_model()
    process_model = get_process_model()

    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp",process_model=process_model)
    image_analysis_toolkit = ImageAnalysisToolkit(model=image_analysis_model)
    video_analysis_toolkit = VideoAnalysisToolkit(working_directory="tmp/video", model=process_model, use_audio_transcription=True)
    audio_analysis_toolkit = AudioAnalysisToolkit(cache_dir="tmp/audio", audio_reasoning_model=audio_reasoning_model)
    code_runner_toolkit = CodeExecutionToolkit(sandbox="subprocess", verbose=True)

    tools=[
            FunctionTool(document_processing_toolkit.extract_document_content),
            FunctionTool(image_analysis_toolkit.ask_question_about_image),
            FunctionTool(audio_analysis_toolkit.ask_question_about_audio),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
            FunctionTool(code_runner_toolkit.execute_code),
        ]

    from camel.agents import ChatAgent

    camel_agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=tools,
        max_iteration=15,
    )

    return camel_agent

async def create_documentation_processing_agent(connected_mcp_toolkit, tools):
    """Create and initialize the documentation_processing_agent."""

    model = get_worker_model()
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
        agent_name="documentation_processing_agent",
        system_message_generator=get_system_message,
        model=model,
        mcp_tool=connected_mcp_toolkit,
        mcp_toolkit=filtered_tools,
        agent_specific_tools=tools,
    )

    print("documentation_processing_agent created successfully with reasoning capabilities.")
    return agent

async def main():
    print("Initializing documentation_processing_agent...")

    agent_id_param = "documentation_processing_agent"
    agent_description = "This agent is a helpful assistant that can process a variety of local and remote documents, including pdf, docx, images, audio, and video, etc."

    # Setup MCP toolkit
    connected_mcp_toolkit = await setup_mcp_toolkit(agent_id_param, agent_description)

    camel_documentation_processing_assistant = await create_documentation_processing_assistant()

    async def documentation_processing_assistant(task: str) -> str:
        """
        An agent-style assistant provides a unified execution interface for web content acquisition, multimodal data analysis, 
        and code execution. It enables agents to retrieve and interpret information across heterogeneous data sources, 
        including URLs, images, audio, and video, and to perform programmatic execution for computation, simulation, or automated processing.

        Args:
            task (str): An instruction or problem to be solved by the agent.

        Returns:
            str: The first message (msg0) from the agent's response.
        """
        camel_documentation_processing_assistant.reset()
        resp = await camel_documentation_processing_assistant.astep(task)
        msg0 = resp.msgs[0]
        # Try to get content if it exists, otherwise convert to str
        if hasattr(msg0, "content"):
            return msg0.content
        return str(msg0)

    tools = [FunctionTool(documentation_processing_assistant)]

    try:
        # Create agent
        agent = await create_documentation_processing_agent(connected_mcp_toolkit, tools)

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
