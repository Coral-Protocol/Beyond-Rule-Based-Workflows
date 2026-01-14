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

from utils.config import get_model, get_worker_model, get_web_model, get_web_planning_model, get_process_model
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
    """Get the system message for the web agent."""
    return f"""
    ===== RULES OF WEB AGENT =====
    You are an advanced `web_agent` powered by web browsing/searching capabilities and working within the Coral server ecosystem, but you are not able to run code script. 

    Core Capabilities:
    1. Web Browsing and Searching 
    - You MUST call `web_assistant` to approach ALL received tasks.
    - You must NOT attempt to analyze or solve these tasks on your own under any circumstances.
    - If you encounter any blocker from calling assistant, you should inform `information_flow_orchestrator` rather than re-try yourself.

    2. Communication with Other Agents  
    - Use `list_agents` only to discover the existence of other agents.
    - You are strictly restricted to communicating via `send_message` with the `information_flow_orchestrator` only.
    - **Always call `send_message` to report your findings to `information_flow_orchestrator` after you after receiving results from `web_assistant`.**
    - You may send messages only to `information_flow_orchestrator`; even when instructed to collaborate with other agents, you must relay everything exclusively through `information_flow_orchestrator` and never contact any other agent directly.
    - **If you have been previously queried by information_flow_orchestrator or asked to perform any task, you MUST first call send_message to respond before calling wait_for_mentions.**
    

    If you encounter a blocker, you should report it to information_flow_orchestrator and recommend to information_flow_orchestrator which agents should be involved (but you must NEVER message those agents directly).


    Here is the message history:
    -- Start of messages and status --
    <resource>http://localhost:5555/devmode/exampleApplication/privkey/{os.getenv("GAIA_TASK_ID")}/sse</resource>
    -- End of messages and status --
    
    """

async def create_web_assistant():

    sys_msg = f"""
            You are a helpful assistant that can search the web, extract webpage content, simulate browser actions, and provide relevant information to solve the given task.
            Keep in mind that:
            - Do not be overly confident in your own knowledge. Searching can provide a broader perspective and help validate existing knowledge.  
            - If one way fails to provide an answer, try other ways or methods. The answer does exists.
            - If the search snippet is unhelpful but the URL comes from an authoritative source, try visit the website for more details.  
            - When looking for specific numerical values (e.g., dollar amounts), prioritize reliable sources and avoid relying only on search snippets.  
            - When solving tasks that require web searches, check Wikipedia first before exploring other websites.  
            - You can also simulate browser actions to get more information or verify the information you have found.
            - Browser simulation is also helpful for finding target URLs. Browser simulation operations do not necessarily need to find specific answers, but can also help find web page URLs that contain answers (usually difficult to find through simple web searches). You can find the answer to the question by performing subsequent operations on the URL, such as extracting the content of the webpage.
            - Do not solely rely on document tools or browser simulation to find the answer, you should combine document tools and browser simulation to comprehensively process web page information. Some content may need to do browser simulation to get, or some content is rendered by javascript.
            - In your response, you should mention the urls you have visited and processed.

            Here are some tips that help you perform web search:
            - Never add too many keywords in your search query! Some detailed results need to perform browser interaction to get, not using search toolkit.
            - If the question is complex, search results typically do not provide precise answers. It is not likely to find the answer directly using search toolkit only, the search query should be concise and focuses on finding official sources rather than direct answers.
            For example, as for the question "What is the maximum length in meters of #9 in the first National Geographic short on YouTube that was ever released according to the Monterey Bay Aquarium website?", your first search term must be coarse-grained like "National Geographic YouTube" to find the youtube website first, and then try other fine-grained search terms step-by-step to find more urls.
            - The results you return do not have to directly answer the original question, you only need to collect relevant information.
            """

    model = get_worker_model()
    web_model = get_web_model()
    web_planning_model = get_web_planning_model()
    process_model = get_process_model()

    search_toolkit = SearchToolkit()
    document_processing_toolkit = DocumentProcessingToolkit(cache_dir="tmp",process_model=process_model)
    video_analysis_toolkit = VideoAnalysisToolkit(working_directory="tmp/video", model=process_model, use_audio_transcription=True)
    browser_simulator_toolkit = AsyncBrowserToolkit(headless=True, cache_dir="tmp/browser", planning_agent_model=web_planning_model, web_agent_model=web_model)

    tools=[
            FunctionTool(search_toolkit.search_google),
            FunctionTool(search_toolkit.search_wiki),
            FunctionTool(search_toolkit.search_wiki_revisions),
            FunctionTool(search_toolkit.search_archived_webpage),
            FunctionTool(document_processing_toolkit.extract_document_content),
            FunctionTool(browser_simulator_toolkit.browse_url),
            FunctionTool(video_analysis_toolkit.ask_question_about_video),
        ]

    from camel.agents import ChatAgent

    camel_agent = ChatAgent(
        system_message=sys_msg,
        model=model,
        tools=tools,
        max_iteration=15,
    )

    return camel_agent

async def create_web_agent(connected_mcp_toolkit, tools):
    """Create and initialize the web agent."""

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
        agent_name="web_agent",
        system_message_generator=get_system_message,
        model=model,
        mcp_tool=connected_mcp_toolkit,
        mcp_toolkit=filtered_tools,
        agent_specific_tools=tools,
    )

    print("web_agent created successfully with reasoning capabilities.")
    return agent

async def main():
    print("Initializing web_agent...")

    agent_id_param = "web_agent"
    agent_description = "This agent is a helpful assistant that can search the web, extract webpage content, and retrieve relevant information."

    # Setup MCP toolkit
    connected_mcp_toolkit = await setup_mcp_toolkit(agent_id_param, agent_description)

    camel_web_assistant = await create_web_assistant()

    async def web_assistant(task: str) -> str:
        """
        An agent-style assistant provides a unified execution interface for web search, historical web content retrieval
        , and video analysis. It enables agents to retrieve, inspect, and interpret 
        information across heterogeneous online sources, including Google Search, Wikipedia and its revision history, 
        archived webpages, and video content.

        Args:
            task (str): An instruction or problem to be solved by the agent.

        Returns:
            str: The first message (msg0) from the agent's response.
        """
        camel_web_assistant.reset()
        resp = await camel_web_assistant.astep(task)
        msg0 = resp.msgs[0]
        # Try to get content if it exists, otherwise convert to str
        if hasattr(msg0, "content"):
            return msg0.content
        return str(msg0)

    tools = [FunctionTool(web_assistant)]

    try:
        # Create agent
        agent = await create_web_agent(connected_mcp_toolkit, tools)

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
        print(f"Error during web_agent operation: {repr(e)}")
    finally:
        if connected_mcp_toolkit:
            await connected_mcp_toolkit.disconnect()
        print(f"Disconnecting {agent_id_param}...")
        print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main()) 
