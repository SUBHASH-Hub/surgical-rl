#!/usr/bin/env python3
"""
Phase 3A — Camera Capture Test
================================
Confirms we can capture a numpy RGB frame from the SOFA endoscopic camera
BEFORE writing the full 50-episode collection script.

Tests four strategies in order of preference, reports shape/dtype for each,
and saves a sample PNG so you can visually verify the frame looks correct.

Run from the repo root (setup_env.sh already sourced):
    python3 scripts/test_camera_capture.py

Expected result:  Frame shape (H, W, 3)  dtype uint8  (likely 480x480 or 600x600)

Author: Subhash Arockiadoss
"""

# ── gym → gymnasium shim (MUST be first, before any env import) ───────────
import sys
import gymnasium
sys.modules['gym'] = gymnasium
sys.modules['gym.spaces'] = gymnasium.spaces
# ─────────────────────────────────────────────────────────────────────────

import os
import time
import traceback
import numpy as np

sys.path.insert(0, '.')   # make sure envs/ is importable from repo root

# ── Colour helpers ────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):    print(f"{GREEN}  ✓  {msg}{RESET}")
def fail(msg):  print(f"{RED}  ✗  {msg}{RESET}")
def info(msg):  print(f"{CYAN}  ·  {msg}{RESET}")
def warn(msg):  print(f"{YELLOW}  !  {msg}{RESET}")

OUTPUT_PNG = "data/rgb_frames/camera_test_frame.png"

# ─────────────────────────────────────────────────────────────────────────
def check_frame(frame, label: str) -> bool:
    """Validate a candidate frame and print diagnostics."""
    if frame is None:
        fail(f"[{label}] returned None")
        return False
    if not isinstance(frame, np.ndarray):
        fail(f"[{label}] type={type(frame).__name__}, expected np.ndarray")
        return False

    info(f"[{label}] shape={frame.shape}  dtype={frame.dtype}  "
         f"min={frame.min()}  max={frame.max()}")

    if frame.ndim != 3 or frame.shape[2] != 3:
        fail(f"[{label}] expected (H,W,3), got {frame.shape}")
        return False

    if frame.dtype != np.uint8:
        warn(f"[{label}] dtype={frame.dtype} — will cast to uint8 in collection script")

    if frame.max() == 0:
        warn(f"[{label}] frame is all-black — SOFA may need more warm-up steps")
    else:
        ok(f"[{label}] non-zero pixel values confirmed ✓")

    return True


def save_png(frame: np.ndarray, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if frame.dtype != np.uint8:
        frame = np.clip(frame, 0, 255).astype(np.uint8)
    try:
        from PIL import Image
        Image.fromarray(frame).save(path)
        ok(f"Saved sample PNG → {path}  (open to visually verify the scene)")
        return
    except ImportError:
        pass
    try:
        import imageio
        imageio.imwrite(path, frame)
        ok(f"Saved sample PNG → {path}  (open to visually verify the scene)")
        return
    except ImportError:
        warn("Neither PIL nor imageio found — skipping PNG save  "
             "(pip install Pillow to enable)")


# ─────────────────────────────────────────────────────────────────────────
def run_test():
    print(f"\n{'='*60}")
    print("  Phase 3A — SOFA Camera Capture Test")
    print(f"{'='*60}\n")

    # ── 1. Import ─────────────────────────────────────────────────────────
    info("Importing TissueRetractionV2 …")
    try:
        from envs.tissue_retraction_v2 import TissueRetractionV2
        ok("Import OK")
    except Exception as e:
        fail(f"Import failed: {e}")
        traceback.print_exc()
        sys.exit(1)

    # ── 2. Create env ─────────────────────────────────────────────────────
    # Use string 'headless' — confirmed working from Phase 3 diagnostic.
    # Falls back to 'human' only if headless raises (e.g. no EGL).
    info("Creating env with render_mode='headless' …")
    env = None
    used_mode = None
    for mode in ('headless', 'human'):
        try:
            env = TissueRetractionV2(env_kwargs={"render_mode": mode})
            ok(f"Env created  (render_mode='{mode}')")
            used_mode = mode
            break
        except Exception as e:
            warn(f"render_mode='{mode}' failed: {e}")

    if env is None:
        fail("Could not create env with any render mode.")
        sys.exit(1)

    # ── 3. Reset ──────────────────────────────────────────────────────────
    info("Calling env.reset() …")
    try:
        result = env.reset()
        obs = result[0] if isinstance(result, tuple) else result
        ok(f"reset() OK — state obs shape: {np.array(obs).shape}")
    except Exception as e:
        fail(f"env.reset() raised: {e}")
        traceback.print_exc()
        env.close()
        sys.exit(1)

    # ── 4. Warm-up steps (give SOFA time to render) ────────────────────────
    info("Running 3 warm-up steps so SOFA renders the scene …")
    try:
        for _ in range(3):
            action = env.action_space.sample() * 0.0   # no-op
            env.step(action)
        ok("Warm-up steps done")
    except Exception as e:
        warn(f"Warm-up step raised (non-fatal): {e}")

    # ─────────────────────────────────────────────────────────────────────
    # FRAME CAPTURE STRATEGIES
    # Tried in order — stop at first success.
    # ─────────────────────────────────────────────────────────────────────
    winning_frame    = None
    winning_strategy = None

    print(f"\n{'-'*50}")
    print("  Testing frame-capture strategies …")
    print(f"{'-'*50}")

    # ── Strategy A: env._env.render() ─────────────────────────────────────
    # sofa_env's render() reads the pyglet/EGL RGB buffer and returns a
    # numpy array when render_mode is not NONE.
    info("Strategy A — env._env.render()")
    try:
        frame_a = env._env.render()
        if check_frame(frame_a, "A: env._env.render()"):
            winning_frame    = frame_a
            winning_strategy = "A"
    except Exception as e:
        fail(f"Strategy A raised: {e}")

    # ── Strategy B: env.render() ──────────────────────────────────────────
    # The wrapper may forward render() to _env transparently.
    if winning_frame is None:
        info("Strategy B — env.render()  (outer wrapper)")
        try:
            frame_b = env.render()
            if check_frame(frame_b, "B: env.render()"):
                winning_frame    = frame_b
                winning_strategy = "B"
        except Exception as e:
            fail(f"Strategy B raised: {e}")

    # ── Strategy C: _maybe_update_rgb_buffer + get_rgb_from_open_gl ───────
    # Direct internal sofa_env buffer pull — works even if render() is None.
    if winning_frame is None:
        info("Strategy C — _maybe_update_rgb_buffer() + get_rgb_from_open_gl()")
        try:
            inner = env._env
            inner._maybe_update_rgb_buffer()
            frame_c = inner.get_rgb_from_open_gl()
            if check_frame(frame_c, "C: get_rgb_from_open_gl()"):
                winning_frame    = frame_c
                winning_strategy = "C"
        except Exception as e:
            fail(f"Strategy C raised: {e}")

    # ── Strategy D: pyglet window buffer ──────────────────────────────────
    # Last resort: capture the pyglet back-buffer directly.
    if winning_frame is None:
        info("Strategy D — pyglet window.get_image_data() buffer")
        try:
            inner = env._env
            win   = getattr(inner, 'window', None) or getattr(inner, '_window', None)
            if win is None:
                raise AttributeError("No 'window' or '_window' attr on env._env")
            win.switch_to()
            win.dispatch_events()
            buf    = win.get_image_data()
            pitch  = buf.width * len(buf.format)
            data   = buf.get_data(buf.format, pitch)
            arr    = np.frombuffer(data, dtype=np.uint8)
            arr    = arr.reshape((buf.height, buf.width, len(buf.format)))
            # pyglet gives RGBA or BGRA — keep first 3 channels, flip vertical
            frame_d = arr[::-1, :, :3]
            if check_frame(frame_d, "D: pyglet buffer"):
                winning_frame    = frame_d
                winning_strategy = "D"
        except Exception as e:
            fail(f"Strategy D raised: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    if winning_frame is not None:
        ok(f"SUCCESS  —  Strategy {winning_strategy} works!")
        ok(f"Frame: shape={winning_frame.shape}  dtype={winning_frame.dtype}")
        save_png(winning_frame, OUTPUT_PNG)

        capture_lines = {
            "A": "frame = env._env.render()",
            "B": "frame = env.render()",
            "C": ("env._env._maybe_update_rgb_buffer()\n"
                  "      frame = env._env.get_rgb_from_open_gl()"),
            "D": "# see Strategy D block in this script for full buffer reshape",
        }
        print(f"\n{GREEN}  ► Capture line to use in collect_rgb_frames.py:{RESET}")
        print(f"      {capture_lines[winning_strategy]}")
        h, w = winning_frame.shape[:2]
        print(f"\n{GREEN}  ► Frame dimensions:{RESET}  {h} x {w} x 3  "
              f"(will resize to 224x224 for CNN input if needed)")

    else:
        fail("ALL strategies failed.")
        print(f"\n{YELLOW}  Diagnostic — render-related attrs on env._env:{RESET}")
        try:
            attrs = [a for a in dir(env._env)
                     if any(k in a.lower()
                            for k in ("render", "rgb", "camera", "image",
                                      "buffer", "window", "pyglet"))]
            for a in attrs:
                try:
                    val = getattr(env._env, a)
                    print(f"    env._env.{a:40s}  {type(val).__name__}")
                except Exception:
                    print(f"    env._env.{a:40s}  <error reading>")
        except Exception:
            pass
        print(f"\n{YELLOW}  Paste this output and we'll debug further.{RESET}")
        env.close()
        sys.exit(1)

    env.close()
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_test()