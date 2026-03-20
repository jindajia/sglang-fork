# Standard library imports
import atexit
import concurrent.futures
import json
import math
import os
import re
import signal
import sys
import threading
import time
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Third-party imports
import docker
import openai
import tqdm
import yaml
from datasets import load_dataset
from fire import Fire

# R2E-Gym imports
import r2egym
from r2egym.agenthub import SUPPORTED_REPOS
from r2egym.agenthub.agent.agent import AgentArgs, Agent
from r2egym.agenthub.environment.env import EnvArgs, RepoEnv
from r2egym.agenthub.openhands.agent.openhands_agent import (
    OpenhandsAgent,
    OpenhandsAgentArgs,
)
from r2egym.agenthub.openhands.environment.openhands_env import OpenhandsEnv
from r2egym.agenthub.runtime.docker import DockerRuntime
from r2egym.agenthub.trajectory import TrajectoryStep, Trajectory
from r2egym.agenthub.utils.log import get_logger
from r2egym.agenthub.utils.utils import get_parsed_commit, match_dockerimage_to_repo
from r2egym.docker_bash_utils.docker_list_tags import fetch_docker_tags


@dataclass
class EvalAgentArgs:
    dataset: str
    split: str
    dataset_num_shards: int = 1
    output_dir: str = "./output"
    exp_name: Optional[str] = None
    dataset_shard_index: int = 0
    max_steps: int = 40
    add_stepcount_message: bool = False
    max_workers: Optional[int] = None
    llm_name: str = "gpt-4o"
    use_existing: bool = True
    skip_successful: bool = False
    disk_prune_threshold_gb: int = 100
    temperature: Optional[float] = 0.6
    top_p: Optional[float] = 0.95
    use_fn_calling: bool = True
    backend: str = "kubernetes"  # "kubernetes" or "docker"
    max_reward_calc_time: int = 300
    env_mode: str = "r2egym"
    prepull_images: bool = False
    pass_n: int = 1  # Number of repetitions per dataset entry
    max_output_tokens: int = 32768  # Max output tokens per LLM query
    max_context_tokens: int = 128000  # Max input context tokens per LLM query
    max_model_tokens: int = 128_000  # Max total (input + output) tokens per LLM query
    max_total_token_limit: int = (
        5_000_000  # 5M tokens Max cumulative tokens across entire run
    )
    max_exec_time: int = 120  # Max execution time per environment step (seconds)
    max_total_time: int = 7200  # Max total time for entire agent run (seconds)
    max_llm_time: int = 3600  # Max time per LLM query (seconds)
    base_url: str = "http://localhost:8000/v1"
    api_key: Optional[str] = None
    dataset_image_field: Optional[str] = (
        None  # Docker image field name, set by env_mode if None
    )


##############################################################################
# Constants
##############################################################################
AGENT_NAME = "OpenhandsAgent"

##############################################################################
# Initialize Logger
##############################################################################
logger = get_logger(__name__)  # Initialize the logger

##############################################################################
# Initialize File Lock for Thread-Safe Writing
##############################################################################
file_lock = threading.Lock()

##############################################################################
# Global Cleanup Registry and Signal Handling
##############################################################################
# Global registry to track active environments for cleanup
active_environments = set()
cleanup_lock = threading.Lock()


def register_environment(env):
    """Register an environment for cleanup tracking"""
    with cleanup_lock:
        active_environments.add(env)


def unregister_environment(env):
    """Unregister an environment from cleanup tracking"""
    with cleanup_lock:
        active_environments.discard(env)


def cleanup_all_environments():
    """Cleanup all registered environments"""
    with cleanup_lock:
        environments_to_cleanup = list(active_environments)

    for env in environments_to_cleanup:
        try:
            logger.info(
                f"Emergency cleanup for environment: {getattr(env, 'ds', {}).get(dataset_image_field, 'unknown')}"
            )
            env.close()
        except Exception as e:
            logger.error(f"Error during emergency cleanup: {e}")

    with cleanup_lock:
        active_environments.clear()


def signal_handler(signum, frame):
    """Handle termination signals gracefully"""
    logger.warning(f"Received signal {signum}, initiating graceful shutdown...")
    cleanup_all_environments()
    sys.exit(1)


# Register signal handlers
signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
signal.signal(signal.SIGTERM, signal_handler)  # Termination signal

# Register atexit cleanup as final fallback
atexit.register(cleanup_all_environments)


##############################################################################
# Utility Function
##############################################################################


def prepull_docker_image(docker_image: str) -> bool:
    """
    Prepulls a single Docker image.

    Args:
        docker_image: The Docker image name to pull

    Returns:
        True if successful, False otherwise
    """
    try:
        client = docker.from_env()
        logger.info(f"Pulling Docker image: {docker_image}")
        client.images.pull(docker_image)
        logger.info(f"Successfully pulled Docker image: {docker_image}")
        return True
    except Exception as e:
        logger.error(f"Failed to pull Docker image {docker_image}: {e}")
        return False


def prepull_docker_images(
    ds_selected: List[Dict],
    max_workers: Optional[int] = None,
    dataset_image_field: str = "docker_image",
) -> None:
    """
    Prepulls all Docker images in parallel before starting the main execution.

    Args:
        ds_selected: List of dataset entries containing docker_image keys
        max_workers: Maximum number of threads for parallel pulling
        dataset_image_field: Docker image field name
    """
    # Extract unique Docker images
    docker_images = list(
        set([ds_entry[dataset_image_field] for ds_entry in ds_selected])
    )
    logger.info(
        f"Starting parallel prepull of {len(docker_images)} unique Docker images..."
    )

    # Use ThreadPoolExecutor for I/O bound operations like Docker pulls
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all pull tasks
        future_to_image = {
            executor.submit(prepull_docker_image, docker_image): docker_image
            for docker_image in docker_images
        }

        # Track results
        successful_pulls = []
        failed_pulls = []

        for future in concurrent.futures.as_completed(future_to_image):
            docker_image = future_to_image[future]
            try:
                success = future.result()
                if success:
                    successful_pulls.append(docker_image)
                else:
                    failed_pulls.append(docker_image)
            except Exception as e:
                logger.error(f"Exception during prepull of {docker_image}: {e}")
                failed_pulls.append(docker_image)

    logger.info(
        f"Prepull completed. Success: {len(successful_pulls)}, Failed: {len(failed_pulls)}"
    )
    if failed_pulls:
        logger.warning(f"Failed to pull images: {failed_pulls}")


def sanitize_name_for_path(name: str) -> str:
    """
    Sanitizes a name to be filesystem-safe by replacing problematic characters.

    Args:
        name: The name to sanitize

    Returns:
        A filesystem-safe version of the name
    """
    return name.replace("/", "_").replace(":", "_").replace(" ", "_").replace(".", "_")


def create_structured_path(
    output_dir: str,
    dataset_name: str,
    agent_name: str,
    model_name: str,
    max_steps: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    run_id: Optional[int] = None,
    instance_id: str = None,
) -> Path:
    """
    Creates the structured directory path following the standard hierarchy.

    Directory structure:
    {output_dir}/{dataset_name}/{agent_name}/{model_name}_max{max_steps}_temp{temperature}_topp{top_p}/run{run_id}/[{instance_id}]

    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        agent_name: Name of the agent
        model_name: Name of the model
        max_steps: Maximum steps configuration
        temperature: LLM temperature setting
        top_p: LLM top_p setting
        run_id: Run number for multiple passes
        instance_id: Optional instance ID name

    Returns:
        Path object representing the structured path
    """
    # Sanitize names for filesystem
    dataset_safe = sanitize_name_for_path(dataset_name)
    agent_safe = sanitize_name_for_path(agent_name)
    model_safe = sanitize_name_for_path(model_name)

    # Build the directory structure
    base_path = Path(output_dir)
    dataset_path = base_path / dataset_safe
    agent_path = dataset_path / agent_safe

    # Create config name with temperature and top_p
    config_name = f"{model_safe}_max{max_steps}"
    if temperature is not None:
        config_name += f"_temp{temperature}"
    if top_p is not None:
        config_name += f"_topp{top_p}"

    config_path = agent_path / config_name

    # Add run_id if provided
    if run_id is not None:
        config_path = config_path / f"run{run_id}"

    if instance_id:
        instance_safe = sanitize_name_for_path(instance_id)
        return config_path / instance_safe

    return config_path


def create_structured_directories(
    output_dir: str,
    dataset_name: str,
    agent_name: str,
    model_name: str,
    max_steps: int,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    run_id: Optional[int] = None,
    instance_id: str = None,
    use_existing: bool = True,
) -> Dict[str, Path]:
    """
    Creates the structured directory hierarchy and returns paths to key directories.

    Directory structure:
    {output_dir}/
    └── {dataset_name}/
        └── {agent_name}/
            └── {model_name}_max{max_steps}_temp{temperature}_topp{top_p}/
                └── run{run_id}/
                    └── {instance_id}/
                        ├── llm_completions/
                        └── logs/

    Args:
        output_dir: Base output directory
        dataset_name: Name of the dataset
        agent_name: Name of the agent (e.g., "Openhands")
        model_name: Name of the model
        max_steps: Maximum steps configuration
        temperature: LLM temperature setting
        top_p: LLM top_p setting
        run_id: Run number for multiple passes
        instance_id: Instance ID name

    Returns:
        Dictionary containing paths to created directories
    """
    # Use the structured path function to generate paths
    base_path = Path(output_dir)
    dataset_path = base_path / sanitize_name_for_path(dataset_name)
    agent_path = dataset_path / sanitize_name_for_path(agent_name)
    config_path = create_structured_path(
        output_dir, dataset_name, agent_name, model_name, max_steps, temperature, top_p
    )
    run_path = create_structured_path(
        output_dir,
        dataset_name,
        agent_name,
        model_name,
        max_steps,
        temperature,
        top_p,
        run_id,
    )
    instance_path = create_structured_path(
        output_dir,
        dataset_name,
        agent_name,
        model_name,
        max_steps,
        temperature,
        top_p,
        run_id,
        instance_id,
    )

    # Sanitize instance ID name for use in directory naming
    instance_safe = sanitize_name_for_path(instance_id)

    # Handle existing directory renaming logic
    if instance_path.exists():
        should_rename = False

        # Check if both trajectory.json and reward.json are valid (used for both use_existing cases)
        trajectory_file = instance_path / "trajectory.json"
        reward_file = instance_path / "reward.json"

        trajectory_valid = False
        reward_valid = False

        # Check trajectory.json
        if trajectory_file.exists():
            try:
                with open(trajectory_file) as f:
                    trajectory_data = json.load(f)
                    if isinstance(trajectory_data, dict):
                        # Try to validate with Trajectory object
                        try:
                            Trajectory.load_from_model_dump_json(
                                json.dumps(trajectory_data)
                            )
                            trajectory_valid = True
                        except:
                            pass
            except (json.JSONDecodeError, Exception):
                pass

        # Check reward.json
        if reward_file.exists():
            try:
                with open(reward_file) as f:
                    reward_data = json.load(f)
                    if isinstance(reward_data, dict):
                        reward_valid = True
            except (json.JSONDecodeError, Exception):
                pass

        if not use_existing:
            # If use_existing is False and directory exists, always rename
            should_rename = True
            rename_as_complete = (
                trajectory_valid and reward_valid
            )  # Check validity even when forcing rename
        else:
            # If use_existing is True, check if both files are valid to decide whether to rename
            if trajectory_valid and reward_valid:
                should_rename = False  # Keep existing directory
                rename_as_complete = False  # Not used when should_rename is False
            else:
                should_rename = True
                rename_as_complete = (
                    trajectory_valid and reward_valid
                )  # Complete if both valid

        if should_rename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if rename_as_complete:
                # Both files exist and are valid - rename as complete
                new_name = f"{instance_safe}_complete_{timestamp}"
            else:
                # Files missing or invalid - rename as incomplete
                new_name = f"{instance_safe}_incomplete_{timestamp}"
            new_path = run_path / new_name
            instance_path.rename(new_path)
            print(f"Renamed existing directory to: {new_path}")

    # Create subdirectories
    llm_completions_path = instance_path / "llm_completions"
    logs_path = instance_path / "logs"

    # Create all directories
    llm_completions_path.mkdir(parents=True, exist_ok=True)
    logs_path.mkdir(parents=True, exist_ok=True)

    return {
        "base": base_path,
        "dataset": dataset_path,
        "agent": agent_path,
        "config": config_path,
        "run": run_path,
        "instance": instance_path,
        "llm_completions": llm_completions_path,
        "logs": logs_path,
    }


##############################################################################
# openhands agent Functions
##############################################################################


def runagent(
    ds,
    run_id: int = 1,
    output_dir: str = "./output",
    dataset_name: str = "unknown",
    exp_name: Optional[str] = None,
    max_steps=40,
    llm_name="gpt-4o",
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    use_fn_calling: bool = True,
    backend: str = "kubernetes",  # "kubernetes" or "docker"
    max_reward_calc_time: int = 300,
    env_mode: str = "r2egym",
    max_output_tokens: int = 32768,
    max_context_tokens: int = 128000,
    max_model_tokens: int = 128000,
    max_total_token_limit: int = 1000000,
    max_exec_time: int = 120,
    max_total_time: int = 7200,
    max_llm_time: int = 3600,
    base_url: str = "http://localhost:8000/v1",
    api_key: Optional[str] = None,
    add_stepcount_message: bool = False,
    use_existing: bool = True,
    max_workers: Optional[int] = None,
    verbose: bool = True,
    dataset_image_field: str = "docker_image",  # Docker image field name
    disk_prune_threshold_gb: int = 100,  # Disk prune threshold in GB
) -> Optional[str]:
    """
    Runs the openhands agent on a specified Docker image.

    Args:
        ds: Dataset entry containing docker_image and other metadata
        run_id: Run number for multiple passes (1, 2, 3, ...)
        output_dir: Base output directory for structured logging
        dataset_name: Name of the dataset being processed
        exp_name: Experiment name. If not provided, a unique name is generated.
        max_output_tokens: Maximum output tokens per LLM query
        max_context_tokens: Maximum input context tokens per LLM query
        max_model_tokens: Maximum total (input + output) tokens per LLM query
        max_total_token_limit: Maximum cumulative tokens across entire agent run
        max_exec_time: Maximum execution time per environment step (seconds)
        max_total_time: Maximum total time for entire agent run (seconds)
        max_llm_time: Maximum time per LLM query (seconds)
        (other args documented as before)
    """
    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Extract instance_id from ds_entry, fallback to docker_image if not present
    instance_id = ds.get("instance_id", ds[dataset_image_field])

    # Create structured directory hierarchy
    agent_name = AGENT_NAME
    dir_structure = create_structured_directories(
        output_dir=output_dir,
        dataset_name=dataset_name,
        agent_name=agent_name,
        model_name=llm_name,
        max_steps=max_steps,
        temperature=temperature,
        top_p=top_p,
        run_id=run_id,
        instance_id=instance_id,
        use_existing=use_existing,
    )

    # Setup logging with new directory structure
    log_file = dir_structure["logs"] / "agent.log"
    logger = get_logger(
        name=sanitize_name_for_path(f"{ds[dataset_image_field]}_run{run_id}"),
        log_file=str(log_file),
        console=True,
        level=logging.INFO,
    )
    logger.info(
        f"Starting openhands agent (Run {run_id}) on Docker image: {ds[dataset_image_field]}"
    )
    logger.info(
        f"Using LLM: {llm_name} | Base URL: {base_url} | Temperature: {temperature} | Top P: {top_p}"
    )
    logger.info(
        f"Max Steps: {max_steps} | Add Stepcount Message: {add_stepcount_message}"
    )

    # Initialize environment arguments
    env_args = EnvArgs(ds=ds)
    env = None

    try:
        # Initialize the RepoEnv
        env = OpenhandsEnv(
            env_args,
            logger=logger,
            backend=backend,
            mode=env_mode,
            disk_prune_threshold_gb=disk_prune_threshold_gb,
        )

        # Register environment for global cleanup tracking
        register_environment(env)

        repo_root = r2egym.__path__[0]

        # load agent args from yaml file
        agent_args = OpenhandsAgentArgs.from_yaml(
            Path(repo_root) / "agenthub/openhands/config/swe_default.yaml"
        )

        # Load the system prompt template from the correct path

        system_prompt_template_path = (
            os.path.join(repo_root, agent_args.system_prompt_template)
            if not os.path.exists(agent_args.system_prompt_template)
            else agent_args.system_prompt_template
        )
        # If agent_args.system_prompt_template is a Path, read its contents as a raw string
        if isinstance(system_prompt_template_path, (str, Path)) and os.path.exists(
            system_prompt_template_path
        ):
            with open(system_prompt_template_path, "r") as f:
                agent_args.system_prompt_template = f.read()

        instance_prompt_template_path = (
            os.path.join(repo_root, agent_args.instance_prompt_template)
            if not os.path.exists(agent_args.instance_prompt_template)
            else agent_args.instance_prompt_template
        )
        # If agent_args.system_prompt_template is a Path, read its contents as a raw string
        if isinstance(instance_prompt_template_path, (str, Path)) and os.path.exists(
            instance_prompt_template_path
        ):
            with open(instance_prompt_template_path, "r") as f:
                agent_args.instance_prompt_template = f.read()

        # set llm base url and api key
        agent_args.llm_name = llm_name
        agent_args.llm_base_url = base_url
        agent_args.llm_api_key = api_key
        agent_args.temperature = temperature
        agent_args.top_p = top_p
        agent_args.add_stepcount_message = add_stepcount_message

        # set token limits
        agent_args.max_output_tokens = max_output_tokens
        agent_args.max_context_tokens = max_context_tokens
        agent_args.max_model_tokens = max_model_tokens

        # Initialize the agent with LLM completions logging directory
        agent = OpenhandsAgent(
            name=AGENT_NAME,
            args=agent_args,
            logger=logger,
            llm_completions_dir=str(dir_structure["llm_completions"]),
            verbose=verbose,
        )

        # Jisen: Enable Preserved Thinking mode for GLM-4.7 agentic tasks (SWE-Bench).
        # extra_body is injected into every litellm.completion() call for this process.
        if "GLM-4.7" in llm_name or "GLM4.7" in llm_name:  # Jisen
            import litellm as _litellm
            _orig_completion = _litellm.completion
            def _patched_completion(*args, **kw):  # Jisen
                kw.setdefault("extra_body", {})
                kw["extra_body"].update({"thinking": {"type": "enabled", "clear_thinking": False}})
                return _orig_completion(*args, **kw)
            _litellm.completion = _patched_completion

        # run agent openhands agent
        logger.warning(f"running agent with {max_steps} max steps (Run {run_id})")
        trajectory = agent.run(
            env,
            max_steps=max_steps,
            max_steps_absolute=max_steps,  # Combined: use same value for both
            use_fn_calling=use_fn_calling,
            max_total_token_limit=max_total_token_limit,
            max_exec_time=max_exec_time,
            max_total_time=max_total_time,
            max_llm_time=max_llm_time,
        )

        # also get the gt outputs
        reward_calc_time = time.time()
        reward, test_output = env.runtime._calculate_reward(
            get_test_output=True, timeout=max_reward_calc_time
        )
        reward_calc_time = time.time() - reward_calc_time

        # update the trajectory object
        trajectory.reward = reward
        trajectory.test_output = test_output
        trajectory.ds = ds
        trajectory.exp_name = exp_name
        trajectory.reward_calc_time = reward_calc_time  # time taken to calculate reward
        logger.warning(
            f"time taken to calculate reward in seconds: {reward_calc_time:.2f}"
        )

        logger.info(
            f"-" * 25
            + f"{ds[dataset_image_field]} (Run {run_id})"
            + "-" * 25
            + "\n\n"
            + "=" * 25
            + f"TEST OUTPUT (last 2000 chars)"
            + "=" * 25
            + "\n\n"
            + f"{test_output[-2000:]}\n\n"
            + "=" * 25
            + f"REWARD"
            + "=" * 25
            + "\n\n"
            + f"{reward}\n"
            + "=" * 50
            + "\n\n",
            file_only=not verbose,
        )

        # Save separate output_patch.json file in the instance directory
        output_patch_data = {
            "docker_image": ds[dataset_image_field],
            "run_id": run_id,
            "output_patch": trajectory.output_patch,
        }
        output_patch_file = dir_structure["instance"] / "output_patch.json"
        with open(output_patch_file, "w") as f:
            json.dump(output_patch_data, f, indent=2)

        logger.info(f"Saved output_patch.json to: {output_patch_file}")

        # Save reward computation information as reward.json
        reward_data = {
            "docker_image": ds[dataset_image_field],
            "run_id": run_id,
            "reward": reward,
            "test_output": test_output,
            "reward_calc_time": reward_calc_time,
            "exp_name": exp_name,
            "ds": ds,
        }
        reward_file = dir_structure["instance"] / "reward.json"
        with open(reward_file, "w") as f:
            json.dump(reward_data, f, indent=2)

        logger.info(f"Saved reward.json to: {reward_file}")

        # Add run_id to trajectory and save trajectory.json in the instance directory
        trajectory_file = dir_structure["instance"] / "trajectory.json"
        trajectory_data = json.loads(trajectory.model_dump_json())
        with open(trajectory_file, "w") as f:
            json.dump(trajectory_data, f, indent=2)

        logger.info(f"Saved trajectory.json to: {trajectory_file}")

        return trajectory.model_dump_json()

    # catch all exceptions
    except Exception as e:
        logger.error(
            f"Error during agent run for {ds[dataset_image_field]} (Run {run_id}): {e}"
        )
        raise e

    finally:
        # Always close the environment and runtime, even if an exception occurs
        try:
            if env is not None:
                logger.info(
                    f"Closing environment for Docker image: {ds[dataset_image_field]} (Run {run_id})"
                )
                env.close()
        except Exception as cleanup_error:
            logger.error(
                f"Error during environment cleanup for {ds[dataset_image_field]} (Run {run_id}): {cleanup_error}"
            )
        finally:
            # Unregister environment from global cleanup tracking
            unregister_environment(env)


def runagent_multiple(args: EvalAgentArgs):
    """
    Runs the openhands agent on a shard of work runs (after filtering existing runs).

    The function creates all work runs (dataset entries × pass_n repetitions), filters out
    existing runs, then shards the remaining work evenly across the specified number of shards.

    Args:
        args: EvalAgentArgs containing all configuration parameters including:
            - dataset_num_shards: Total number of shards to divide work across
            - dataset_shard_index: Index of the shard to process (0-based)
            - pass_n: Number of repetitions per Docker image (in 11112222333 order)
            - output_dir: Base directory for structured output
            - exp_name: Experiment name for the JSONL file
            - max_steps: Maximum steps for the agent run
            - max_workers: Maximum number of threads to use
            - prepull_images: Whether to prepull Docker images in parallel before starting execution
    """
    # Extract arguments
    dataset_num_shards = args.dataset_num_shards
    dataset_shard_index = args.dataset_shard_index
    pass_n = args.pass_n
    exp_name = args.exp_name
    output_dir = args.output_dir
    max_steps = args.max_steps
    llm_name = args.llm_name
    temperature = args.temperature
    top_p = args.top_p
    use_fn_calling = args.use_fn_calling
    backend = args.backend
    max_reward_calc_time = args.max_reward_calc_time
    env_mode = args.env_mode
    max_output_tokens = args.max_output_tokens
    max_context_tokens = args.max_context_tokens
    max_model_tokens = args.max_model_tokens
    max_total_token_limit = args.max_total_token_limit
    max_exec_time = args.max_exec_time
    max_total_time = args.max_total_time
    max_llm_time = args.max_llm_time
    max_workers = args.max_workers
    prepull_images = args.prepull_images
    use_existing = args.use_existing
    skip_successful = args.skip_successful
    add_stepcount_message = args.add_stepcount_message
    dataset_image_field = args.dataset_image_field
    disk_prune_threshold_gb = args.disk_prune_threshold_gb

    # Validate parameter combination
    assert (
        not skip_successful or use_existing
    ), "skip_successful can only be used when use_existing is True"

    # Load the dataset
    ds = load_dataset(args.dataset, split=args.split, num_proc=16)
    logger.info(f"Total dataset size: {len(ds)}, Num shards: {dataset_num_shards}, Shard index: {dataset_shard_index}")

    # shuffle the dataset
    ds = ds.shuffle(seed=42)

    # Create ds_selected with repetitions for the entire dataset in 11112222333 order
    ds_selected_with_runs = []
    for ds_entry in ds:
        for run_num in range(1, pass_n + 1):
            ds_selected_with_runs.append((ds_entry, run_num))

    logger.info(
        f"Dataset: {args.dataset}, Split: {args.split}, Num_total: {len(ds)}, pass_n: {pass_n}"
    )
    logger.info(
        f"Created {len(ds_selected_with_runs)} total runs ({len(ds)} Docker images with {pass_n} passes each)."
    )

    # Generate a unique experiment name if not provided
    if exp_name is None:
        exp_name = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create the output directory structure (at the dataset level for trajectories)
    config_path = create_structured_path(
        output_dir, args.dataset, AGENT_NAME, llm_name, max_steps, temperature, top_p
    )
    config_path.mkdir(parents=True, exist_ok=True)

    # Set dataset_image_field based on env_mode if not provided
    if dataset_image_field is None:
        if env_mode in ["swebench", "r2egym", "swerebench"]:
            dataset_image_field = "docker_image"
        elif env_mode in ["swesmith", "r2egym"]:
            dataset_image_field = "image_name"
        else:
            raise ValueError(f"Invalid environment mode: {env_mode}")

    # Docker-specific trajectory files will be created per docker image and run

    if use_existing:
        existing_entries = []
        for ds_entry, run_id in ds_selected_with_runs:
            # Extract instance_id from ds_entry, fallback to docker_image field if not present
            instance_id = ds_entry.get("instance_id", ds_entry[dataset_image_field])
            instance_safe = sanitize_name_for_path(instance_id)
            run_dir = config_path / f"run{run_id}"
            instance_dir = run_dir / instance_safe
            json_file = instance_dir / "trajectory.json"

            if json_file.exists():
                try:
                    with open(json_file) as f:
                        trajectory_data = json.load(f)
                        # Validate JSON structure
                        if not isinstance(trajectory_data, dict):
                            print(
                                f"Invalid trajectory.json format for {instance_id} run{run_id}: not a dict"
                            )
                            continue
                        # Convert dict back to Trajectory object for validation
                        trajectory_obj = Trajectory.load_from_model_dump_json(
                            json.dumps(trajectory_data)
                        )
                        existing_entries.append((instance_id, run_id))
                except json.JSONDecodeError as e:
                    print(
                        f"Invalid JSON in trajectory.json for {instance_id} run{run_id}: {e}"
                    )
                except Exception as e:
                    print(
                        f"Error processing trajectory.json for {instance_id} run{run_id}: {e}"
                    )

        ds_selected_with_runs = [
            (ds_entry, run_id)
            for ds_entry, run_id in ds_selected_with_runs
            if (ds_entry.get("instance_id", ds_entry[dataset_image_field]), run_id)
            not in existing_entries
        ]

    if skip_successful:
        existing_entries = []
        for ds_entry, run_id in ds_selected_with_runs:
            # Extract instance_id from ds_entry, fallback to docker_image field if not present
            instance_id = ds_entry.get("instance_id", ds_entry[dataset_image_field])
            instance_safe = sanitize_name_for_path(instance_id)
            run_dir = config_path / f"run{run_id}"
            instance_dir = run_dir / instance_safe
            reward_file = instance_dir / "reward.json"

            if reward_file.exists():
                try:
                    with open(reward_file) as f:
                        reward_data = json.load(f)
                        # Validate JSON structure
                        if not isinstance(reward_data, dict):
                            print(
                                f"Invalid reward.json format for {instance_id} run{run_id}: not a dict"
                            )
                            continue
                        if reward_data.get("reward") == 1:
                            existing_entries.append((instance_id, run_id))
                except json.JSONDecodeError as e:
                    print(
                        f"Invalid JSON in reward.json for {instance_id} run{run_id}: {e}"
                    )
                except Exception as e:
                    print(
                        f"Error processing reward.json for {instance_id} run{run_id}: {e}"
                    )

        ds_selected_with_runs = [
            (ds_entry, run_id)
            for ds_entry, run_id in ds_selected_with_runs
            if (ds_entry.get("instance_id", ds_entry[dataset_image_field]), run_id)
            not in existing_entries
        ]

    logger.info(
        f"Starting openhands agent on {len(ds_selected_with_runs)} total runs after filtering."
    )

    # Apply sharding to ds_selected_with_runs
    total_runs = len(ds_selected_with_runs)
    shard_size = math.ceil(total_runs / dataset_num_shards)
    shard_start = dataset_shard_index * shard_size
    shard_end = min(shard_start + shard_size, total_runs)
    ds_selected_with_runs = ds_selected_with_runs[shard_start:shard_end]
    
    logger.info(
        f"Sharding: Processing shard {dataset_shard_index}/{dataset_num_shards} (size={shard_size}, range={shard_start}:{shard_end}) = {len(ds_selected_with_runs)} runs from {total_runs} total runs."
    )

    # Prepull all Docker images in parallel before starting main execution
    if ds_selected_with_runs and prepull_images:
        logger.info("Prepulling Docker images before starting main execution...")
        # Extract unique ds_entries for prepulling (no need to prepull same image multiple times)
        unique_ds_entries = list(
            {
                ds_entry[dataset_image_field]: ds_entry
                for ds_entry, _ in ds_selected_with_runs
            }.values()
        )
        prepull_docker_images(
            unique_ds_entries,
            max_workers=max_workers,
            dataset_image_field=dataset_image_field,
        )
        logger.info("Docker image prepull completed.")

    def process_result(result, docker_image_run, pbar):
        """Helper function to process a single task result"""
        # The trajectory.json file is now saved directly in runagent function
        # This function now only handles progress bar updates
        pbar.set_description(f"Completed: {docker_image_run}")
        pbar.update(1)

    if max_workers and max_workers > 1:
        # Use ProcessPoolExecutor for parallel execution
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all tasks to the executor using keyword arguments
            future_to_image = {
                executor.submit(
                    runagent,
                    ds_entry,
                    run_id=run_id,
                    output_dir=output_dir,
                    dataset_name=args.dataset,
                    exp_name=exp_name,
                    max_steps=max_steps,
                    llm_name=llm_name,
                    temperature=temperature,
                    top_p=top_p,
                    add_stepcount_message=add_stepcount_message,
                    use_fn_calling=use_fn_calling,
                    backend=backend,
                    max_reward_calc_time=max_reward_calc_time,
                    env_mode=env_mode,
                    max_output_tokens=max_output_tokens,
                    max_context_tokens=max_context_tokens,
                    max_model_tokens=max_model_tokens,
                    max_total_token_limit=max_total_token_limit,
                    max_exec_time=max_exec_time,
                    max_total_time=max_total_time,
                    max_llm_time=max_llm_time,
                    base_url=args.base_url,
                    api_key=args.api_key,
                    use_existing=use_existing,
                    verbose=False,
                    dataset_image_field=dataset_image_field,
                    disk_prune_threshold_gb=disk_prune_threshold_gb,
                ): f"{ds_entry.get('instance_id', ds_entry[dataset_image_field])}_run{run_id}"
                for ds_entry, run_id in ds_selected_with_runs
            }

            try:
                with tqdm.tqdm(
                    total=len(future_to_image), desc="Running agent"
                ) as pbar:
                    for future in concurrent.futures.as_completed(future_to_image):
                        docker_image_run = future_to_image[future]
                        try:
                            result = future.result()
                            process_result(result, docker_image_run, pbar)
                        except Exception as e:
                            logger.error(f"Exception for {docker_image_run}: {e}")
                            pbar.set_description(f"Failed: {docker_image_run}")
                            pbar.update(1)
                            continue
            except KeyboardInterrupt:
                logger.warning(
                    "Received interrupt signal, cancelling remaining tasks..."
                )
                # Cancel all pending futures
                for future in future_to_image.keys():
                    if not future.done():
                        future.cancel()
                # Wait for running tasks to complete cleanup
                logger.warning("Waiting for running tasks to complete cleanup...")
                raise
            except Exception as e:
                logger.error(f"Unexpected error in ProcessPoolExecutor: {e}")
                # Cancel all pending futures on unexpected errors
                for future in future_to_image.keys():
                    if not future.done():
                        future.cancel()
                raise
    else:
        # Sequential execution for single worker - no need for ProcessPoolExecutor overhead
        try:
            with tqdm.tqdm(
                total=len(ds_selected_with_runs), desc="Running agent"
            ) as pbar:
                for ds_entry, run_id in ds_selected_with_runs:
                    docker_image_run = f"{ds_entry.get('instance_id', ds_entry[dataset_image_field])}_run{run_id}"
                    pbar.set_description(f"Processing: {docker_image_run}")

                    try:
                        result = runagent(
                            ds_entry,
                            run_id=run_id,
                            output_dir=output_dir,
                            dataset_name=args.dataset,
                            exp_name=exp_name,
                            max_steps=max_steps,
                            llm_name=llm_name,
                            temperature=temperature,
                            top_p=top_p,
                            add_stepcount_message=add_stepcount_message,
                            use_fn_calling=use_fn_calling,
                            backend=backend,
                            max_reward_calc_time=max_reward_calc_time,
                            env_mode=env_mode,
                            max_output_tokens=max_output_tokens,
                            max_context_tokens=max_context_tokens,
                            max_model_tokens=max_model_tokens,
                            max_total_token_limit=max_total_token_limit,
                            max_exec_time=max_exec_time,
                            max_total_time=max_total_time,
                            max_llm_time=max_llm_time,
                            base_url=args.base_url,
                            api_key=args.api_key,
                            use_existing=use_existing,
                            verbose=True,
                            dataset_image_field=dataset_image_field,
                            disk_prune_threshold_gb=disk_prune_threshold_gb,
                        )
                        process_result(result, docker_image_run, pbar)
                    except Exception as e:
                        logger.error(f"Exception for {docker_image_run}: {e}")
                        raise e
        except KeyboardInterrupt:
            logger.warning(
                "Received interrupt signal, stopping sequential execution..."
            )
            raise

    logger.info(
        f"openhands agent completed on {len(ds_selected_with_runs)} total runs."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run openhands agent on multiple Docker images."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./output",
        help="Base directory for structured output logs and trajectories.",
    )
    parser.add_argument(
        "--max_workers",
        type=int,
        default=None,
        help="Maximum number of parallel workers.",
    )
    parser.add_argument(
        "--dataset_shard_index", type=int, default=0, help="Shard index for work runs (0-based)."
    )
    parser.add_argument("--dataset_num_shards", type=int, default=1, help="Total number of shards to split work across.")
    parser.add_argument(
        "--pass_n",
        type=int,
        default=1,
        help="Number of repetitions per dataset entry (11112222333 order).",
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name.")
    parser.add_argument("--split", type=str, default="test", help="Dataset split.")
    parser.add_argument("--llm_name", type=str, default="gpt-4o", help="LLM name.")
    parser.add_argument(
        "--base_url",
        type=str,
        default="http://localhost:8000/v1",
        help="Base URL for LLM API.",
    )
    parser.add_argument(
        "--use_fn_calling",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Use function calling (True/False).",
    )
    parser.add_argument("--exp_name", type=str, default=None, help="Experiment name.")
    parser.add_argument(
        "--temperature", type=float, default=None, help="LLM temperature."
    )
    parser.add_argument("--top_p", type=float, default=0.95, help="LLM top_p.")
    parser.add_argument(
        "--max_steps", type=int, default=40, help="Maximum steps for agent."
    )
    parser.add_argument(
        "--add_stepcount_message",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Add stepcount message to the agent (True/False).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="docker",
        help="Backend to use: 'kubernetes' or 'docker'.",
    )
    parser.add_argument(
        "--max_reward_calc_time",
        type=int,
        default=300,
        help="Max reward calculation time.",
    )

    parser.add_argument(
        "--env_mode", type=str, default="r2egym", help="Environment mode type."
    )
    parser.add_argument(
        "--max_output_tokens",
        type=int,
        default=32768,
        help="Max output tokens per LLM query.",
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=128000,
        help="Max input context tokens per LLM query.",
    )
    parser.add_argument(
        "--max_model_tokens",
        type=int,
        default=128000,
        help="Max total (input + output) tokens per LLM query.",
    )
    parser.add_argument(
        "--max_total_token_limit",
        type=int,
        default=1_000_000_000,
        help="Max cumulative tokens across entire agent run. Default is 1 billion tokens. Useful for saving cost.",
    )
    parser.add_argument(
        "--max_exec_time",
        type=int,
        default=120,
        help="Max execution time per environment step (seconds).",
    )
    parser.add_argument(
        "--max_total_time",
        type=int,
        default=7200,
        help="Max total time for entire agent run (seconds).",
    )
    parser.add_argument(
        "--max_llm_time",
        type=int,
        default=3600,
        help="Max time per LLM query (seconds).",
    )
    parser.add_argument(
        "--prepull_images",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Prepull Docker images (True/False).",
    )
    parser.add_argument(
        "--api_key", type=str, default=None, help="API key for LLM service."
    )
    parser.add_argument(
        "--use_existing",
        type=lambda x: (str(x).lower() == "true"),
        default=True,
        help="Use existing trajectories (True/False).",
    )
    parser.add_argument(
        "--skip_successful",
        type=lambda x: (str(x).lower() == "true"),
        default=False,
        help="Skip existing trajectories (True/False).",
    )
    parser.add_argument(
        "--disk_prune_threshold_gb",
        type=int,
        default=100,
        help="Disk prune threshold in GB.",
    )
    parser.add_argument(
        "--dataset_image_field",
        type=str,
        default=None,
        help="Docker image field name (set by env_mode if None).",
    )
    args = parser.parse_args()

    # Convert argparse.Namespace to dict and filter only EvalAgentArgs fields
    arg_dict = vars(args)
    runagent_args_fields = {field for field in EvalAgentArgs.__dataclass_fields__}
    filtered_args = {k: v for k, v in arg_dict.items() if k in runagent_args_fields}
    agent_args = EvalAgentArgs(**filtered_args)

    runagent_multiple(agent_args)
