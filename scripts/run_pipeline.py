#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

import yaml


# ============================================================
# PATHS
# ============================================================

ROOT = Path("/workspace/plow")
PIPELINE_FILE = ROOT / "pipeline.yaml"

WORKTREE_ROOT = Path("/workspace/worktrees")

PIPELINE_ROOT = ROOT / "pipeline"
LOG_DIR = PIPELINE_ROOT / "logs"
STATE_DIR = PIPELINE_ROOT / "state"


# ============================================================
# COLORS / OUTPUT
# ============================================================

def log(message=""):
    print(message, flush=True)


def section(title):
    log()
    log("=" * 80)
    log(title)
    log("=" * 80)


# ============================================================
# COMMAND EXECUTION
# ============================================================

def run_command(
    command,
    cwd=None,
    check=True,
    capture=True,
    env=None,
):
    """
    Run a command and stream output.
    """

    command = [str(x) for x in command]

    log()
    log("$ " + " ".join(command))

    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.STDOUT if capture else None,
        text=True,
        bufsize=1,
        universal_newlines=True,
        env=env,
    )

    output = []

    if process.stdout is not None:
        for line in process.stdout:
            print(line, end="", flush=True)
            output.append(line)

    return_code = process.wait()

    result = "".join(output)

    if check and return_code != 0:
        raise RuntimeError(
            f"Command failed with exit code {return_code}:\n"
            f"{' '.join(command)}"
        )

    return return_code, result


def git(worktree, *args, check=True):
    return run_command(
        ["git", *args],
        cwd=worktree,
        check=check,
        capture=True,
    )


# ============================================================
# GIT HELPERS
# ============================================================

def current_branch(worktree):
    _, output = git(
        worktree,
        "branch",
        "--show-current",
    )

    return output.strip()


def latest_commit(worktree):
    _, output = git(
        worktree,
        "rev-parse",
        "HEAD",
    )

    return output.strip()


def git_status(worktree):
    _, output = git(
        worktree,
        "status",
        "--porcelain",
    )

    return output.strip()


def assert_clean(worktree):
    status = git_status(worktree)

    if status:
        raise RuntimeError(
            f"Worktree contains uncommitted changes:\n"
            f"{worktree}\n\n"
            f"{status}"
        )


def assert_branch(worktree, expected):
    actual = current_branch(worktree)

    if actual != expected:
        raise RuntimeError(
            f"Wrong branch for {worktree}\n"
            f"Expected: {expected}\n"
            f"Actual:   {actual}"
        )

    log(f"✓ branch: {actual}")


def assert_git_repository(worktree):
    _, output = git(
        worktree,
        "rev-parse",
        "--is-inside-work-tree",
    )

    if output.strip() != "true":
        raise RuntimeError(
            f"Not a git worktree: {worktree}"
        )


# ============================================================
# FILE HELPERS
# ============================================================

def assert_file(path, description):
    if not path.exists():
        raise RuntimeError(
            f"Missing {description}:\n"
            f"{path}"
        )

    if path.is_file() and path.stat().st_size == 0:
        raise RuntimeError(
            f"{description} is empty:\n"
            f"{path}"
        )

    log(f"✓ {description}: {path}")


def assert_artifacts(worktree, artifacts):
    for artifact in artifacts:
        assert_file(
            worktree / artifact,
            f"artifact {artifact}",
        )


# ============================================================
# WORKTREE VALIDATION
# ============================================================

def validate_worktree(name, config):
    section(f"CHECKING WORKTREE: {name}")

    worktree = Path(config["worktree"])
    branch = config["branch"]

    if not worktree.exists():
        raise RuntimeError(
            f"Worktree does not exist:\n"
            f"{worktree}"
        )

    if not worktree.is_dir():
        raise RuntimeError(
            f"Worktree is not a directory:\n"
            f"{worktree}"
        )

    assert_git_repository(worktree)
    assert_branch(worktree, branch)

    return worktree


# ============================================================
# PROMPT VALIDATION
# ============================================================

def load_prompt(config):
    prompt_path = ROOT / config["prompt"]

    assert_file(
        prompt_path,
        "prompt",
    )

    prompt = prompt_path.read_text()

    if not prompt.strip():
        raise RuntimeError(
            f"Prompt is empty:\n"
            f"{prompt_path}"
        )

    return prompt


# ============================================================
# TOOL VALIDATION
# ============================================================

def command_exists(command):
    result = subprocess.run(
        ["which", command],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    return result.returncode == 0


def check_tools():
    section("CHECKING REQUIRED TOOLS")

    for command in [
        "git",
        "nix",
        "agent",
    ]:
        if not command_exists(command):
            raise RuntimeError(
                f"Required command not found: {command}"
            )

        result = subprocess.run(
            ["which", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        log(
            f"✓ {command}: "
            f"{result.stdout.strip()}"
        )


# ============================================================
# STATE
# ============================================================

def state_file(agent_name):
    return STATE_DIR / f"{agent_name}.done"


def mark_complete(agent_name, commit):
    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    state_file(agent_name).write_text(
        commit + "\n"
    )


def is_complete(agent_name):
    return state_file(agent_name).exists()


# ============================================================
# BUILD AGENT COMMAND
# ============================================================

def build_agent_command(prompt):
    """
    IMPORTANT:

    We deliberately do NOT use:

        --workspace

    because subprocess cwd is already the
    target worktree.

    Every agent runs inside:

        nix develop

    and then:

        agent -p --trust <prompt>
    """

    return [
        "agent",
        "-p",
        "--trust",
        prompt,
    ]


# ============================================================
# RUN SINGLE AGENT
# ============================================================

def run_agent(name, config):
    section(f"STARTING {name}")

    worktree = validate_worktree(
        name,
        config,
    )

    prompt = load_prompt(
        config
    )

    # --------------------------------------------------------
    # Optional parent branch
    # --------------------------------------------------------

    parent = config.get("parent")

    if parent:
        log()
        log(
            f"Synchronizing {name} "
            f"with parent: {parent}"
        )

        assert_clean(worktree)

        git(
            worktree,
            "merge",
            "--no-edit",
            parent,
        )

    # --------------------------------------------------------
    # Log setup
    # --------------------------------------------------------

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    log_file = (
        LOG_DIR /
        f"{name}.log"
    )

    log_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Agent command
    # --------------------------------------------------------

    command = build_agent_command(
        prompt
    )

    log()
    log("Worktree:")
    log(f"  {worktree}")

    log()
    log("Branch:")
    log(f"  {current_branch(worktree)}")

    log()
    log("Prompt:")
    log(f"  {ROOT / config['prompt']}")

    log()
    log("Log:")
    log(f"  {log_file}")

    log()
    log("COMMAND:")
    log(
        "  nix develop --command "
        "agent -p --trust <prompt>"
    )

    log()
    log("Launching Cursor Agent...")
    log()

    # --------------------------------------------------------
    # Environment
    # --------------------------------------------------------

    env = os.environ.copy()

    env["PYTHONUNBUFFERED"] = "1"

    # --------------------------------------------------------
    # Launch
    # --------------------------------------------------------

    with log_file.open(
        "w",
        buffering=1,
    ) as logfile:

        process = subprocess.Popen(
            command,
            cwd=str(worktree),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        for line in process.stdout:

            # Terminal
            print(
                line,
                end="",
                flush=True,
            )

            # Log
            logfile.write(line)
            logfile.flush()

        return_code = process.wait()

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    if return_code != 0:

        raise RuntimeError(
            f"{name} Cursor Agent failed "
            f"with exit code {return_code}"
        )

    log()
    log(
        f"✓ {name} Cursor Agent finished"
    )

    # --------------------------------------------------------
    # Artifact validation
    # --------------------------------------------------------

    section(
        f"VALIDATING {name} ARTIFACTS"
    )

    assert_artifacts(
        worktree,
        config.get(
            "artifacts",
            [],
        ),
    )

    # --------------------------------------------------------
    # Git validation
    # --------------------------------------------------------

    section(
        f"VALIDATING {name} GIT STATE"
    )

    status = git_status(
        worktree
    )

    if status:
        log()
        log("Agent produced changes:")
        log(status)

        git(
            worktree,
            "add",
            "-A",
        )

        git(
            worktree,
            "commit",
            "-m",
            f"pipeline: complete {name}",
        )

        log(
            f"✓ {name} changes committed by pipeline"
        )
    else:
        log(
            f"✓ {name} produced no uncommitted changes"
        )

    commit = latest_commit(worktree)

    log(
        f"✓ {name} commit: {commit}"
    )


    # --------------------------------------------------------
    # State
    # --------------------------------------------------------

    mark_complete(
        name,
        commit,
    )

    section(
        f"{name} COMPLETE"
    )

    log(
        f"Commit: {commit}"
    )


# ============================================================
# RUN PARALLEL AGENTS
# ============================================================

def run_parallel(agent_configs):
    section(
        "STARTING PARALLEL OPTIMIZATION AGENTS"
    )

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    processes = {}

    # --------------------------------------------------------
    # Start all agents
    # --------------------------------------------------------

    for name, config in agent_configs.items():

        worktree = validate_worktree(
            name,
            config,
        )

        prompt = load_prompt(
            config
        )

        parent = config.get("parent")

        if parent:
            assert_clean(worktree)

            git(
                worktree,
                "merge",
                "--no-edit",
                parent,
            )

        command = build_agent_command(
            prompt
        )

        log_file = (
            LOG_DIR /
            f"{name}.log"
        )

        log()
        log(
            f"Launching {name}"
        )

        log(
            f"  worktree: {worktree}"
        )

        log(
            f"  branch:   {current_branch(worktree)}"
        )

        log(
            f"  log:      {log_file}"
        )

        logfile = log_file.open(
            "w",
            buffering=1,
        )

        env = os.environ.copy()

        env["PYTHONUNBUFFERED"] = "1"

        process = subprocess.Popen(
            command,
            cwd=str(worktree),
            stdout=logfile,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )

        processes[name] = {
            "process": process,
            "logfile": logfile,
            "config": config,
            "worktree": worktree,
        }

        log(
            f"  PID: {process.pid}"
        )

    # --------------------------------------------------------
    # Wait
    # --------------------------------------------------------

    failures = []

    for name, info in processes.items():

        process = info["process"]
        logfile = info["logfile"]
        config = info["config"]
        worktree = info["worktree"]

        return_code = process.wait()

        logfile.close()

        log()
        log(
            f"{name} exited with code "
            f"{return_code}"
        )

        if return_code != 0:

            failures.append(
                name
            )

            continue

        try:

            section(
                f"VALIDATING {name}"
            )

            assert_artifacts(
                worktree,
                config.get(
                    "artifacts",
                    [],
                ),
            )

            status = git_status(
                worktree
            )

            if status:
                raise RuntimeError(
                    f"{name} left "
                    f"uncommitted changes:\n"
                    f"{status}"
                )

            commit = latest_commit(
                worktree
            )

            mark_complete(
                name,
                commit,
            )

            log(
                f"✓ {name} completed"
            )

            log(
                f"  commit: {commit}"
            )

        except Exception as exc:

            log()
            log(
                f"✗ {name} validation failed"
            )

            log(str(exc))

            failures.append(
                name
            )

    if failures:

        raise RuntimeError(
            "Parallel agents failed: "
            + ", ".join(failures)
        )

    section(
        "PARALLEL OPTIMIZATION COMPLETE"
    )


# ============================================================
# INTEGRATION
# ============================================================

def integrate_branch(branch):
    section(
        f"INTEGRATING {branch}"
    )

    assert_clean(ROOT)

    git(
        ROOT,
        "merge",
        "--no-ff",
        "--no-edit",
        branch,
    )

    log(
        f"✓ integrated {branch}"
    )


def integrate():
    section("INTEGRATING OPTIMIZATION RESULTS")

    # Kernel first
    integrate_branch(
        "agent/kernels"
    )

    # Runtime second
    integrate_branch(
        "agent/runtime"
    )

    log()
    log(
        "✓ kernel + runtime optimizations integrated"
    )


# ============================================================
# VALIDATION WORKTREE
# ============================================================

def prepare_validation_worktree(
    config
):
    section(
        "PREPARING FINAL VALIDATION WORKTREE"
    )

    worktree = validate_worktree(
        "agent6",
        config,
    )

    assert_clean(
        worktree
    )

    final_commit = latest_commit(
        ROOT
    )

    log()
    log(
        f"Final integrated commit: "
        f"{final_commit}"
    )

    # Bring validation branch to final integrated state.
    git(
        worktree,
        "merge",
        "--no-edit",
        final_commit,
    )

    log(
        "✓ validation worktree synchronized"
    )


# ============================================================
# DRY RUN
# ============================================================

def dry_run(pipeline):
    section(
        "PIPELINE DRY RUN"
    )

    # --------------------------------------------------------
    # Repository
    # --------------------------------------------------------

    log()
    log("Repository")
    log("-" * 80)

    if not ROOT.exists():
        raise RuntimeError(
            f"Repository does not exist: {ROOT}"
        )

    if not (
        ROOT / ".git"
    ).exists():
        raise RuntimeError(
            f"Not a git repository: {ROOT}"
        )

    log(
        f"✓ repository: {ROOT}"
    )

    # --------------------------------------------------------
    # YAML
    # --------------------------------------------------------

    section(
        "VALIDATING PIPELINE CONFIGURATION"
    )

    for key in [
        "version",
        "project",
        "agents",
    ]:
        if key not in pipeline:
            raise RuntimeError(
                f"Missing YAML field: {key}"
            )

        log(
            f"✓ YAML field: {key}"
        )

    agents = pipeline["agents"]

    required = [
        "agent1",
        "agent2",
        "agent3",
        "optimization",
        "agent6",
    ]

    for name in required:

        if name not in agents:
            raise RuntimeError(
                f"Missing pipeline stage: {name}"
            )

        log(
            f"✓ pipeline stage: {name}"
        )

    # --------------------------------------------------------
    # Agent configuration
    # --------------------------------------------------------

    all_agents = {
        "agent1": agents["agent1"],
        "agent2": agents["agent2"],
        "agent3": agents["agent3"],
        "agent6": agents["agent6"],
    }

    optimization_agents = (
        agents[
            "optimization"
        ][
            "agents"
        ]
    )

    all_agents.update(
        optimization_agents
    )

    # --------------------------------------------------------
    # Prompts
    # --------------------------------------------------------

    section(
        "CHECKING PROMPTS"
    )

    for name, config in all_agents.items():

        load_prompt(
            config
        )

        log(
            f"✓ {name} prompt"
        )

    # --------------------------------------------------------
    # Worktrees
    # --------------------------------------------------------

    section(
        "CHECKING WORKTREES"
    )

    for name, config in all_agents.items():

        validate_worktree(
            name,
            config,
        )

    # --------------------------------------------------------
    # Artifacts
    # --------------------------------------------------------

    section(
        "CHECKING ARTIFACT CONFIGURATION"
    )

    for name, config in all_agents.items():

        artifacts = config.get(
            "artifacts",
            [],
        )

        log(
            f"✓ {name}: "
            f"{len(artifacts)} artifacts"
        )

        for artifact in artifacts:
            log(
                f"    - {artifact}"
            )

    # --------------------------------------------------------
    # Tools
    # --------------------------------------------------------

    check_tools()

    # --------------------------------------------------------
    # Nix validation
    # --------------------------------------------------------

    section(
        "CHECKING NIX DEVELOPMENT ENVIRONMENT"
    )

    nix_result = subprocess.run(
        [
            "nix",
            "develop",
            "--command",
            "bash",
            "-c",
            "echo NIX_DEV_OK",
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    print(
        nix_result.stdout,
        end="",
        flush=True,
    )

    if nix_result.returncode != 0:
        raise RuntimeError(
            "nix develop failed"
        )

    if "NIX_DEV_OK" not in nix_result.stdout:
        raise RuntimeError(
            "nix develop did not return "
            "expected output"
        )

    log(
        "✓ nix develop works"
    )

    # --------------------------------------------------------
    # Execution graph
    # --------------------------------------------------------

    section(
        "PIPELINE EXECUTION GRAPH"
    )

    log(
        """
Agent 1
  │
  ▼
Agent 2
  │
  ▼
Agent 3
  │
  ├───────────────┐
  ▼               ▼
Agent 4         Agent 5
Kernels         Runtime
  │               │
  └───────┬───────┘
          ▼
      Integration
          │
          ▼
       Agent 6
          │
       ┌──┴──┐
       ▼     ▼
     PASS   FAIL
       │
       ▼
      DONE
"""
    )

    # --------------------------------------------------------
    # Result
    # --------------------------------------------------------

    section(
        "DRY RUN PASSED"
    )

    log(
        "No Cursor agents were started."
    )

    log(
        "No source files were modified."
    )

    log(
        "No branches were merged."
    )

    log(
        "No benchmarks were executed."
    )


# ============================================================
# LOAD YAML
# ============================================================

def load_pipeline():
    if not PIPELINE_FILE.exists():
        raise RuntimeError(
            f"Missing pipeline.yaml:\n"
            f"{PIPELINE_FILE}"
        )

    try:

        data = yaml.safe_load(
            PIPELINE_FILE.read_text()
        )

    except yaml.YAMLError as exc:

        raise RuntimeError(
            f"Invalid pipeline.yaml:\n{exc}"
        )

    if not isinstance(data, dict):
        raise RuntimeError(
            "pipeline.yaml must contain "
            "a YAML mapping"
        )

    return data


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Qwen ASR optimization pipeline"
        )
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate pipeline without "
            "starting Cursor agents"
        ),
    )

    args = parser.parse_args()

    pipeline = load_pipeline()

    # --------------------------------------------------------
    # Dry run
    # --------------------------------------------------------

    if args.dry_run:

        dry_run(
            pipeline
        )

        return

    # --------------------------------------------------------
    # Directories
    # --------------------------------------------------------

    LOG_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    STATE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    agents = pipeline["agents"]

    # --------------------------------------------------------
    # Agent 1
    # --------------------------------------------------------

    #run_agent(
     #   "agent1",
      #  agents["agent1"],
   # )

    # --------------------------------------------------------
    # Agent 2
    # --------------------------------------------------------

    run_agent(
        "agent2",
        agents["agent2"],
    )

    # --------------------------------------------------------
    # Agent 3
    # --------------------------------------------------------

    run_agent(
        "agent3",
        agents["agent3"],
    )

    # --------------------------------------------------------
    # Agent 4 + Agent 5
    # --------------------------------------------------------

    run_parallel(
        agents[
            "optimization"
        ][
            "agents"
        ]
    )

    # --------------------------------------------------------
    # Integration
    # --------------------------------------------------------

    integrate()

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    prepare_validation_worktree(
        agents["agent6"]
    )

    run_agent(
        "agent6",
        agents["agent6"],
    )

    # --------------------------------------------------------
    # Final validation report
    # --------------------------------------------------------

    validation_worktree = Path(
        agents["agent6"]["worktree"]
    )

    validation_report = (
        validation_worktree /
        "docs/final-validation.md"
    )

    assert_file(
        validation_report,
        "final validation report",
    )

    report = (
        validation_report.read_text()
    )

    if "RESULT: PASS" not in report:

        section(
            "PIPELINE FINISHED — VALIDATION FAILED"
        )

        raise RuntimeError(
            "Agent 6 did not report "
            "RESULT: PASS"
        )

    # --------------------------------------------------------
    # Success
    # --------------------------------------------------------

    section(
        "PIPELINE SUCCESS"
    )

    log(
        "RESULT: PASS"
    )

    log()
    log(
        "Qwen ASR optimization passed "
        "the final validation."
    )


# ============================================================
# ENTRYPOINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        log()
        log(
            "Pipeline interrupted by user."
        )

        sys.exit(130)

    except Exception as exc:

        log()
        log("=" * 80)
        log("PIPELINE FAILED")
        log("=" * 80)
        log()
        log(str(exc))

        sys.exit(1)
