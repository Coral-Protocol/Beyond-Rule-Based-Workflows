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

from utils.config import get_model, get_process_model, get_image_model, get_audio_model
from utils.agent_factory import create_agent, setup_mcp_toolkit, run_agent_loop
from dotenv import load_dotenv
import os
import re
import json
import html
from datetime import datetime
from xml.dom import minidom
import xmltodict



# Suppress parameter description warnings from toolkits
warnings.filterwarnings("ignore", message="Parameter description is missing.*", category=UserWarning)


# Model Configuration optimized for reasoning and problem solving
MODEL_CONFIG = {
   # "max_completion_tokens": 8192,
    "reasoning_effort": "high",
    # Note: O3_MINI doesn't support temperature, top_p, etc. due to reasoning model limitations
}

def is_file_or_folder(path: str):
    base = os.path.basename(path)
    name, ext = os.path.splitext(base)
    if ext:  
        return "file"
    else:
        return "folder"

def get_system_message() -> str:
    """Get the system message for the information_flow_orchestrator."""
    return f"""
    ===== RULES OF INFORMATION FLOW ORCHESTRATOR =====
    You are an advanced `information_flow_orchestrator`. 

    Core Responsibilities:
    1. Inquiry and Relay Management  
    - MONITOR the task process and make sure its reliability and healthy.

    - INQUIRE EXACTLY ONE proper agent at a time to perform any additional reasoning required to support progress or resolve uncertainty.
        (Reasoning here refers to the processes by which task-level instructions are derived, and may include, but is not limited to, **planning, critique, verifification, assess the reliability of intermediate process, replanning, reflection, multi-agent debate, questioning or critique**. 

    - RELAY task-level content and necessarily previous results to EXACTLY ONE proper agent at a time, and only if the task content is explicitly and verbatim produced by other agents.
        When you relay task instructions, DO NOT forward too many instructions at once; always relay them in appropriately small and manageable units.
    
    - Confirm the generated answer with the planner or proper agent(s) and reach a consistent consensus before sumbitted.

    For the purpose of this role, the following distinctions apply:

    - Task Instruction:
        A task instruction is any content that specifies, implies, or guides
        what concrete actions should be taken, by whom, or in what order,
        to advance or complete the task.
        This includes, but is not limited to:
        • directing an agent to retrieve information, inspect artifacts, browse documents, or run tools
        • assigning responsibilities to specific agents

    - Reasoning:
        **Reasoning** refers exclusively to cognitive processes used to derive, evaluate, compare, or reconsider task instructions.
        These processes include, but are not limited to, **planning, critique, verification, assessing the reliability of intermediate results, replanning, reflection, multi-agent debate, questioning, and critique**.
        Reasoning does **not** involve executing actions, retrieving information, inspecting documents, browsing the web, calling APIs, or operating tools.
        It exists solely to produce or assess task instructions, **not to carry them out**.

    2. Communication with Other Agents  
    - Use `list_agents` to become aware of the existence of other agents.  
    - Use `create_thread` to organize communication for each task or topic, and close threads when they become too crowded or inactive.  
        Agents not in a thread cannot see the messages in that thread.  
        Messages in closed threads are no longer visible to any agent, although summaries remain accessible.  
    - Use `send_message` to communicate with other agents.
        You MUST send a message to EXACTLY ONE agent at a time.
        You MUST NOT send messages to multiple agents in a single step or in parallel.
        After sending a message to an agent, you MUST NOT send any further messages—neither to the same agent nor to any other agent—until a reply is received.
    - Use `wait_for_mentions` to receive messages from other agents.

    3. Submit Final Answer
    - Confirm the final answer with the planner or proper agent(s) and reach a consistent consensus.
    - Call submit_answer_tool to submit the final answer when it is generated and verified.

    Here is the message history:
    -- Start of messages and status --
    <resource>http://localhost:5555/devmode/exampleApplication/privkey/{os.getenv("GAIA_TASK_ID")}/sse</resource>
    -- End of messages and status --

    """

async def create_information_flow_orchestrator(connected_mcp_toolkit, tools):
    """Create and initialize the Information flow orchestrator."""

    model = get_model()
    print("Model created successfully")

    mcp_tools = connected_mcp_toolkit.get_tools() if connected_mcp_toolkit else []

    ALLOWED_TOOLS = {
        "create_thread",
        "add_participant",
        "remove_participant",
        "close_thread",
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
        agent_name="information_flow_orchestrator",
        system_message_generator=get_system_message,
        model=model,
        mcp_tool=connected_mcp_toolkit,
        mcp_toolkit=filtered_tools,
        agent_specific_tools=tools,
    )

    print("Information flow orchestrator created successfully with reasoning capabilities.")
    return agent

async def main():
    print("Initializing Information flow orchestrator...")

    agent_id_param = "information_flow_orchestrator"
    agent_description = "This agent is a helpful assistant that can inquire, route information, evaluate partial results, and verify their reliability across multi-agents."

    # Setup MCP toolkit
    connected_mcp_toolkit = await setup_mcp_toolkit(agent_id_param, agent_description)

    async def submit_answer_tool(task: str, raw_answer: str) -> str:
        """
        A toolkit to submit the final answer.

        Args:
            task (str): The original task exactly as provided, without any modification, rewriting, or summarization.
            raw_answer (str): The final answer generated for the given task.

        Returns:
            str: The result of the answer submission.
        """

        # define answer agent
        answer_prompt = f"""
        I am solving a question:
        <question>
        {task}
        </question>

        Now, the raw answer is follow:
        <raw answer>
        {raw_answer}
        <raw answer>

        Now, I need you to determine the final answer. Do not try to solve the question, just pay attention to ONLY the format in which the answer is presented. DO NOT CHANGE THE MEANING OF THE PRIMARY ANSWER.
        You should first analyze the answer format required by the question and then output the final answer that meets the format requirements. 
        Here are the requirements for the final answer:
        <requirements>
        The final answer must be output exactly in the format specified by the question. The final answer should be a number OR as few words as possible OR a comma separated list of numbers and/or strings.
        If you are asked for a number, don't use comma to write your number neither use units such as $ or percent sign unless specified otherwise. Numbers do not need to be written as words, but as digits.
        If you are asked for a string, don't use articles, neither abbreviations (e.g. for cities), and write the digits in plain text unless specified otherwise. In most times, the final string is as concise as possible (e.g. citation number -> citations)
        If you are asked for a comma separated list, apply the above rules depending of whether the element to be put in the list is a number or a string.
        If the the raw_answer is "Give up", just output "Give up".
        </requirements>

        Please output with the final answer according to the requirements without any other text. If the primary answer is already a final answer with the correct format, just output the primary answer.
        """

        # get worker model
        model = get_process_model()

        from camel.agents import ChatAgent

        camel_agent = ChatAgent(
            system_message="You are a helpful assistant that can answer questions and provide final answers.",
            model=model,
        )

        # get final answer
        camel_agent.reset()
        resp = await camel_agent.astep(answer_prompt)
        msg0 = resp.msgs[0]
        final_answer = msg0.content if hasattr(msg0, "content") else str(msg0)

        if raw_answer == "Give up":
            final_answer = "Give up"

        # get resources
        url = f"http://localhost:5555/devmode/exampleApplication/privkey/{os.getenv("GAIA_TASK_ID")}/sse"
        match = re.match(r"http://([^/]+)/(.+)", url)
        server_name, resource_path = match.groups()

        client = None
        for c in connected_mcp_toolkit.clients:
            if c.config.url.startswith(f"http://{server_name}"):
                client = c
                break

        resources_dump = {}

        if client is not None:
            session = client.session
            resources_list = await session.list_resources()
            uri_list = [r.uri for r in resources_list.resources]

            for uri in uri_list:
                contents_result = await session.read_resource(uri)

                raw_xml = contents_result.contents[0].text

                try:
                    structured = xmltodict.parse(raw_xml)
                    resources_dump[str(uri)] = structured
                except Exception:
                    resources_dump[str(uri)] = {"raw_xml": raw_xml}

        # prepare record
        record = {
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "coral_answer": final_answer,
            "resources": resources_dump,
        }

        # save to JSON
        save_path = "submitted_answers.json"

        if os.path.exists(save_path):
            with open(save_path, "r", encoding="utf-8") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(record)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return final_answer
    
    LIFECYCLE_TIMER = {
        "is_started": False,
        "start_time": None,
        "end_time": None,
        "duration_minutes": 40,
    }

    async def start_lifecycle_timer_tool(duration_minutes: int = 40) -> dict:
        """
        A toolkit to start a lifecycle countdown timer.

        This tool is called ONCE at the beginning of a task lifecycle.
        It records the start time and computes the end time. The timer does NOT
        actively count down; it only stores timestamps so other agents can query
        remaining time.

        Args:
            duration_minutes (int):
                Number of minutes for the lifecycle countdown (default: 40).

        Returns:
            dict: {
                "status": "started",
                "start_time": <ISO string>,
                "end_time": <ISO string>,
                "duration_minutes": <int>
            }
        """

        from datetime import datetime, timedelta

        global LIFECYCLE_TIMER

        start_time = datetime.now()
        end_time = start_time + timedelta(minutes=duration_minutes)

        LIFECYCLE_TIMER = {
            "is_started": True,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_minutes": duration_minutes,
        }

        return {
            "status": "started",
            "start_time": LIFECYCLE_TIMER["start_time"],
            "end_time": LIFECYCLE_TIMER["end_time"],
            "duration_minutes": duration_minutes,
        }
    
    await start_lifecycle_timer_tool(60)

    async def query_lifecycle_timer_tool(dummy: str = "ignore") -> str:
        """
        A toolkit to query the remaining lifecycle time.

        Args:
            dummy (str, optional): A no-op parameter to stabilize LLM tool-calling behavior.
                                The value is ignored. Default is "ignore".

        Returns:
            str: JSON string, e.g.
                '{"is_started": true, "remaining_minutes": 10,
                "remaining_seconds": 600, "time_up": false}'
        """

        from datetime import datetime
        import json
        global LIFECYCLE_TIMER

        # Ignore dummy — this ensures LLM always has a schema to follow,
        # preventing null/empty argument generation.
        _ = dummy

        if not LIFECYCLE_TIMER["is_started"]:
            result = {
                "is_started": False,
                "remaining_minutes": None,
                "remaining_seconds": None,
                "time_up": False,
            }
            return json.dumps(result)

        now = datetime.now()
        end_time = datetime.fromisoformat(LIFECYCLE_TIMER["end_time"])
        remaining_sec = int((end_time - now).total_seconds())

        if remaining_sec <= 0:
            result = {
                "is_started": True,
                "remaining_minutes": 0,
                "remaining_seconds": 0,
                "time_up": True,
            }
            return json.dumps(result)

        result = {
            "is_started": True,
            "remaining_minutes": remaining_sec // 60,
            "remaining_seconds": remaining_sec,
            "time_up": False,
        }
        return json.dumps(result)

    '''tools = [FunctionTool(submit_answer_tool),FunctionTool(query_lifecycle_timer_tool)]'''
    tools = [FunctionTool(submit_answer_tool)]

    try:
        # Create agent
        agent = await create_information_flow_orchestrator(connected_mcp_toolkit, tools)

        task = os.getenv("GAIA_TASK_QUESTION")
        file_name = os.getenv("GAIA_TASK_FILE")

        if not task:
            raise ValueError("GAIA_TASK_QUESTION is not set in environment variables")

        print("✅ Loaded task from environment:")
        print(task)

        if is_file_or_folder(file_name) == "file":
            initial_prompt = f"The task you need to approach is {task}, and the related file name is {file_name}, call list_agents to discovery other agents then create a thread with **ALL agents (NEVER MISS ANY ONE)** then start working."
        else:
            initial_prompt = f"The task you need to approach is {task}, call list_agents to discovery other agents then create a thread with all agents then start working."
       
        # Run agent loop
        await run_agent_loop(
            agent=agent,
            agent_id=agent_id_param,
            initial_prompt=initial_prompt,
            loop_prompt="Continue managing the infomration flow, verify the findings and intermediate results, and determine whether a reliable answer has been produced. If yes, submit it; if not, keep going.",
            max_iterations=100,
            sleep_time=1
        )

    except Exception as e:
        print(f"Error during Information flow orchestrator operation: {repr(e)}")
    finally:
        if connected_mcp_toolkit:
            await connected_mcp_toolkit.disconnect()
        print(f"Disconnecting {agent_id_param}...")
        print("Disconnected.")

if __name__ == "__main__":
    asyncio.run(main()) 
