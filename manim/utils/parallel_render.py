"""Parallel rendering for manim using the 'Render Trick' algorithm.

Phase 1: Run construct() with skip_animations to capture scene state before each play().
Phase 2: Render each animation in a separate process via multiprocessing.
Phase 3: Combine partial movie files into the final video.
"""

from __future__ import annotations

import copy
import multiprocessing
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

try:
    import dill
except ImportError:
    raise ImportError(
        "The --parallel flag requires the 'dill' package. "
        "Install it with: pip install dill"
    )

from .. import config, logger
from ..animation.animation import Wait
from ..utils.file_ops import guarantee_existence

if TYPE_CHECKING:
    from ..scene.scene import Scene

__all__ = ["render_parallel"]


@dataclass
class PlayCapture:
    """One captured play() call."""

    index: int
    combined_state_bytes: bytes  # dill-serialized {mobjects, foreground_mobjects, animations}
    partial_file_path: str


# ---------------------------------------------------------------------------
# Phase 1: Capture
# ---------------------------------------------------------------------------


def capture_play_calls(scene: Scene) -> list[PlayCapture]:
    """Run construct() in skip mode, capturing state before each play().

    Returns a list of PlayCapture objects, one per play() call.
    """
    captures: list[PlayCapture] = []
    original_play = scene.play

    # Determine partial movie directory for output file naming
    file_writer = scene.renderer.file_writer
    if not hasattr(file_writer, "partial_movie_directory"):
        raise RuntimeError(
            "Parallel rendering requires write_to_movie=True. "
            "Cannot determine partial movie directory."
        )
    partial_movie_dir = file_writer.partial_movie_directory

    def capturing_play(*args: Any, **kwargs: Any) -> None:
        index = len(captures)

        # Build animations from raw args (converts _AnimationBuilder → Animation).
        # This is the same call that compile_animation_data() makes internally.
        # compile_animations is side-effect free on the scene.
        built_animations = scene.compile_animations(*args, **kwargs)

        # Deepcopy scene state AND animations TOGETHER so that shared references
        # between scene.mobjects and animation.mobject are preserved in the copy.
        combined = copy.deepcopy(
            {
                "mobjects": scene.mobjects,
                "foreground_mobjects": scene.foreground_mobjects,
                "animations": built_animations,
            }
        )

        partial_file_path = str(
            partial_movie_dir
            / f"parallel_{index:05}{config['movie_file_extension']}"
        )

        capture = PlayCapture(
            index=index,
            combined_state_bytes=dill.dumps(combined),
            partial_file_path=partial_file_path,
        )
        captures.append(capture)

        # Run the REAL play() in skip mode so scene state advances correctly.
        # compile_animations will be called again inside, which is safe.
        original_play(*args, **kwargs)

    # Enable skip mode on the renderer (same mechanism as -s flag)
    old_original_skip = scene.renderer._original_skipping_status
    old_skip = scene.renderer.skip_animations
    scene.renderer._original_skipping_status = True
    scene.renderer.skip_animations = True

    # Monkey-patch play and run construct
    scene.play = capturing_play  # type: ignore[assignment]
    try:
        scene.construct()
    except Exception:
        raise
    finally:
        scene.play = original_play  # type: ignore[assignment]
        scene.renderer._original_skipping_status = old_original_skip
        scene.renderer.skip_animations = old_skip

    return captures


# ---------------------------------------------------------------------------
# Phase 2: Worker
# ---------------------------------------------------------------------------


def _serialize_config() -> dict[str, Any]:
    """Serialize the current config to a dict safe for cross-process transfer."""
    result: dict[str, Any] = {}
    for k, v in config._d.items():
        if isinstance(v, Path):
            result[k] = str(v)
        elif hasattr(v, "value"):  # enum (e.g. RendererType)
            result[k] = v.value
        else:
            result[k] = v
    return result


def render_one_animation(worker_args_bytes: bytes) -> str:
    """Worker function: render a single animation to a partial movie file.

    Receives all arguments as a single dill-serialized blob to bypass
    pickle limitations in multiprocessing's spawn context.
    """
    import dill

    worker_args = dill.loads(worker_args_bytes)
    config_dict: dict[str, Any] = worker_args["config_dict"]
    combined_state_bytes: bytes = worker_args["combined_state_bytes"]
    partial_file_path: str = worker_args["partial_file_path"]
    index: int = worker_args["index"]
    source_file_parent: str = worker_args["source_file_parent"]
    scene_class_name: str = worker_args["scene_class_name"]
    scene_module_name: str = worker_args["scene_module_name"]

    # Ensure the source file's directory is importable
    if source_file_parent not in sys.path:
        sys.path.insert(0, source_file_parent)

    # Restore config in this worker process.
    # Use __setitem__ (goes through property setters) for type-safe restoration.
    from manim import config as worker_config

    for k, v in config_dict.items():
        try:
            worker_config[k] = v
        except Exception:
            try:
                worker_config._d[k] = v
            except Exception:
                pass
    worker_config["disable_caching"] = True

    # Deserialize combined state (mobjects + pre-built animations)
    combined = dill.loads(combined_state_bytes)

    # Import the scene class and create a fresh instance (gets its own renderer)
    import importlib

    module = importlib.import_module(scene_module_name)
    SceneClass = getattr(module, scene_class_name)

    scene = SceneClass()
    scene.setup()

    # Restore mobject state from snapshot
    scene.mobjects = combined["mobjects"]
    scene.foreground_mobjects = combined["foreground_mobjects"]

    # Use pre-built animations (already converted from _AnimationBuilder)
    animations = combined["animations"]
    scene.animations = animations
    scene.add_mobjects_from_animations(animations)
    scene.last_t = 0
    scene.stop_condition = None
    scene.moving_mobjects = []
    scene.static_mobjects = []
    scene.duration = scene.get_run_time(animations)

    # Handle static Wait
    if len(animations) == 1 and isinstance(animations[0], Wait):
        if not scene.should_update_mobjects():
            animations[0].is_static_wait = True

    renderer = scene.renderer
    file_writer = renderer.file_writer

    # Ensure target directory exists
    guarantee_existence(Path(partial_file_path).parent)

    # Open stream directly to the target partial file
    file_writer.open_partial_movie_stream(file_path=partial_file_path)

    scene.begin_animations()
    renderer.save_static_frame_data(scene, scene.static_mobjects)

    if scene.is_current_animation_frozen_frame():
        renderer.update_frame(scene, mobjects=scene.moving_mobjects)
        renderer.freeze_current_frame(scene.duration)
    else:
        scene.play_internal()

    file_writer.close_partial_movie_stream()

    return partial_file_path


def run_parallel_workers(
    captures: list[PlayCapture],
    scene_class_name: str,
    scene_module_name: str,
    source_file_parent: str,
    max_workers: int | None = None,
) -> list[str]:
    """Dispatch captured animations to worker processes.

    Returns the ordered list of partial movie file paths.
    """
    if max_workers is None:
        max_workers = max(1, (os.cpu_count() or 2) - 1)

    config_dict = _serialize_config()

    # Serialize all worker args as dill blobs (bypasses pickle for spawn)
    worker_blobs: list[bytes] = []
    for capture in captures:
        blob = dill.dumps(
            {
                "config_dict": config_dict,
                "combined_state_bytes": capture.combined_state_bytes,
                "partial_file_path": capture.partial_file_path,
                "index": capture.index,
                "source_file_parent": source_file_parent,
                "scene_class_name": scene_class_name,
                "scene_module_name": scene_module_name,
            }
        )
        worker_blobs.append(blob)

    ctx = multiprocessing.get_context("spawn")

    partial_files: list[str] = []
    with ctx.Pool(processes=min(max_workers, len(captures))) as pool:
        results = pool.map(render_one_animation, worker_blobs)
        partial_files = list(results)

    return partial_files


# ---------------------------------------------------------------------------
# Phase 3: Combine
# ---------------------------------------------------------------------------


def render_parallel(scene: Scene, scene_class: type, file_path: Path) -> None:
    """Main entry point for parallel rendering.

    Called from commands.py when --parallel is set.
    """
    logger.info(f"Parallel rendering: {scene_class.__name__}")

    # Phase 1: Capture
    logger.info("Phase 1: Capturing animation states...")
    scene.setup()
    captures = capture_play_calls(scene)

    if not captures:
        logger.info("No animations to render.")
        scene.tear_down()
        return

    logger.info(f"Captured {len(captures)} animation(s).")

    # Compute module import path for workers
    source_file = Path(file_path).resolve()
    source_file_parent = str(source_file.parent)
    scene_module_name = source_file.stem
    scene_class_name = scene_class.__name__

    # Phase 2: Parallel render
    logger.info(
        f"Phase 2: Rendering {len(captures)} animation(s) in parallel..."
    )
    partial_files = run_parallel_workers(
        captures,
        scene_class_name,
        scene_module_name,
        source_file_parent,
    )

    # Phase 3: Combine
    logger.info("Phase 3: Combining partial files...")
    file_writer = scene.renderer.file_writer

    # Set partial_movie_files so combine_to_movie() picks them up
    file_writer.partial_movie_files = partial_files
    if file_writer.sections:
        file_writer.sections[-1].partial_movie_files = partial_files

    file_writer.combine_to_movie()

    scene.tear_down()

    logger.info(
        f"Rendered {scene_class.__name__}\n"
        f"Played {len(captures)} animations (parallel)"
    )
