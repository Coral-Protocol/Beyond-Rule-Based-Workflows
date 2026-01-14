import os
import json
import time
import subprocess
from datetime import datetime
from pathlib import Path
from collections import OrderedDict
import threading
import asyncio
import re
import urllib.request
import xmltodict
import sys

from utils.agent_factory import setup_mcp_toolkit
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent # run.py
REPO_ROOT = THIS_DIR.parent # agent_defined_workflow/

GAIA_ROOT = THIS_DIR / "data" / "gaia" / "2023" / "validation"
GAIA_FILE = GAIA_ROOT / "metadata.jsonl"
SUBMITTED_ANS = THIS_DIR / "submitted_answers.json"
CORAL_SERVER_DIR = REPO_ROOT / "coral-server"

AGENT_SCRIPTS = [
    "web_agent.py",
    "planning_agent.py",
    "information_flow_orchestrator.py",
    "document_processing_agent.py",
    "coding_agent.py",
]

# =========================
# Task Timer
# =========================
class TaskTimer:
    def __init__(self, timeout_sec=1800):
        self.timeout = timeout_sec
        self.start_time = None

    def start(self):
        self.start_time = time.time()

    def expired(self):
        return time.time() - self.start_time > self.timeout


# =========================
# Utils
# =========================
def normalize(s):
    return " ".join(str(s).lower().split())


def read_jsonl(path):
    with open(path, "r", encoding="utf8") as f:
        return [json.loads(l) for l in f]


def load_answers():
    if not os.path.exists(SUBMITTED_ANS):
        return []
    try:
        with open(SUBMITTED_ANS, "r", encoding="utf8") as f:
            return json.load(f)
    except Exception:
        return []


def stream_output(name, p):
    def _pipe(pipe, tag):
        for line in iter(pipe.readline, ""):
            if line.strip():
                print(f"[{name}][{tag}] {line.rstrip()}")
        pipe.close()

    threading.Thread(target=_pipe, args=(p.stdout, "OUT"), daemon=True).start()
    threading.Thread(target=_pipe, args=(p.stderr, "ERR"), daemon=True).start()


# =========================
# Coral Server
# =========================
def start_coral_server():
    print("🚀 Starting Coral Server...")
    '''p = subprocess.Popen(
        ["./gradlew", "run"],
        cwd=CORAL_SERVER_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )
    stream_output("coral_server", p)
    return p'''

    return subprocess.Popen(
        ["./gradlew", "run"],
        cwd=CORAL_SERVER_DIR,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def stop_process(p, name):
    if p and p.poll() is None:
        p.terminate()
        try:
            p.wait(timeout=5)
        except Exception:
            p.kill()
    print(f"🛑 {name} stopped")


# =========================
# SSE readiness
# =========================
def _ping(url):
    req = urllib.request.Request(url, method="GET")
    with urllib.request.urlopen(req, timeout=5) as r:
        return r.status


async def wait_for_sse_ready(url, timeout=120):
    print("⏳ Waiting for Coral SSE to be ready...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            await asyncio.to_thread(_ping, url)
            print("✅ Coral SSE ready")
            return
        except Exception:
            await asyncio.sleep(2)
    raise TimeoutError("Coral SSE not ready")

async def setup_mcp_with_retry(agent_id, desc, retries=5):
    last_exc = None
    for i in range(retries):
        try:
            return await setup_mcp_toolkit(agent_id, desc)
        except asyncio.CancelledError:
            last_exc = "CancelledError"
        except Exception as e:
            last_exc = repr(e)

        print(f"⚠️ MCP setup failed, retry {i+1}/{retries}: {last_exc}")
        await asyncio.sleep(3)

    raise RuntimeError(f"MCP setup failed after retries: {last_exc}")

# =========================
# MCP Resources Dump
# =========================
async def get_resources(connected_mcp_toolkit):
    task_id = os.getenv("GAIA_TASK_ID", "")
    url = f"http://localhost:5555/devmode/exampleApplication/privkey/{task_id}/sse"

    match = re.match(r"http://([^/]+)/(.+)", url)
    if not match:
        return {}

    server_name, _ = match.groups()

    client = None
    for c in connected_mcp_toolkit.clients:
        if c.config.url.startswith(f"http://{server_name}"):
            client = c
            break

    if client is None:
        return {}

    session = client.session
    resources_dump = {}

    resources_list = await session.list_resources()
    for r in resources_list.resources:
        try:
            content = await session.read_resource(r.uri)
            raw = content.contents[0].text
            try:
                resources_dump[str(r.uri)] = xmltodict.parse(raw)
            except Exception:
                resources_dump[str(r.uri)] = {"raw_xml": raw}
        except Exception as e:
            resources_dump[str(r.uri)] = {"error": repr(e)}

    return resources_dump


# =========================
# Agents
# =========================
def start_agents():
    procs = []
    for s in AGENT_SCRIPTS:
        print(f"🚀 Starting {s} ...")
        p = subprocess.Popen(
            ["python", "-u", s],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        stream_output(s.replace(".py", ""), p)
        procs.append(p)
    return procs


def stop_agents(procs):
    for p in procs:
        stop_process(p, "agent")
    print("🛑 All agents stopped")


# =========================
# Wait for answer
# =========================
async def wait_for_answer(prev_len, timer):
    print("⏳ Waiting for agents to finish task...")
    while True:
        await asyncio.sleep(5)
        if timer.expired():
            return None, "timeout"
        cur = load_answers()
        if len(cur) > prev_len:
            return cur[-1], "normal"


# =========================
# Main
# =========================
async def main():
    tasks = read_jsonl(GAIA_FILE)
    Path(SUBMITTED_ANS).touch(exist_ok=True)

    for idx, task in enumerate(tasks):
        print("\n============================")
        print(f"🧠 Running GAIA Task {idx+1}/{len(tasks)}")
        print(f"Task ID: {task['task_id']}")
        print(f"Question: {task['Question']}")
        print("============================")

        if any(r.get("task_id") == task["task_id"] for r in load_answers()):
            print("⚠️ Already done, skip")
            continue

        timer = TaskTimer()
        timer.start()
        start_ts = datetime.now().isoformat()

        os.environ["GAIA_TASK_ID"] = task["task_id"]
        os.environ["GAIA_TASK_QUESTION"] = task["Question"]
        os.environ["GAIA_TASK_FILE"] = os.path.join(GAIA_ROOT, task["file_name"])

        coral = None
        agents = None
        toolkit = None
        resources = None

        try:
            coral = start_coral_server()
            sse_url = f"http://localhost:5555/devmode/exampleApplication/privkey/{task['task_id']}/sse"
            await wait_for_sse_ready(sse_url)

            agents = start_agents()

            toolkit = await setup_mcp_with_retry(
                "information_flow_orchestrator",
                "Central orchestrator agent"
            )


            before = len(load_answers())
            record, reason = await wait_for_answer(before, timer)

            try:
                resources = await get_resources(toolkit)
            except Exception as e:
                resources = {"error": repr(e)}

            all_ans = load_answers()

            if record is None:
                new = OrderedDict(
                    start_timestamp=start_ts,
                    finish_timestamp=datetime.now().isoformat(),
                    task_id=task["task_id"],
                    task_level=task["Level"],
                    task=task["Question"],
                    Final_answer=task.get("Final answer"),
                    coral_answer="Give up",
                    Correct=False,
                    terminate_reason=reason,
                    resources=resources,
                )
                all_ans.append(new)
            else:
                gold = task.get("Final answer")
                pred = record.get("coral_answer")
                correct = normalize(gold) == normalize(pred)

                new = OrderedDict(
                    start_timestamp=start_ts,
                    finish_timestamp=datetime.now().isoformat(),
                    task_id=task["task_id"],
                    task_level=task["Level"],
                    task=task["Question"],
                    Final_answer=gold,
                    coral_answer=pred,
                    Correct=correct,
                    resources=resources,
                )
                all_ans[-1] = new

            with open(SUBMITTED_ANS, "w", encoding="utf8") as f:
                json.dump(all_ans, f, indent=2, ensure_ascii=False)

            print("🎉 Task finished")

        finally:
            if toolkit:
                await toolkit.disconnect()
            if agents:
                stop_agents(agents)
            if coral:
                stop_process(coral, "Coral server")

    print("\n🎉 All GAIA tasks finished")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except asyncio.CancelledError:
        print("❌ run.py cancelled at top level, exiting with code 1")
        sys.exit(1)
