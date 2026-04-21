import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")
    requests_stub.post = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("requests.post should be stubbed in tests")
    )
    sys.modules["requests"] = requests_stub

if "matplotlib" not in sys.modules:
    matplotlib_stub = types.ModuleType("matplotlib")
    pyplot_stub = types.ModuleType("pyplot")
    pyplot_stub.subplots = lambda *args, **kwargs: (_ for _ in ()).throw(
        AssertionError("matplotlib plotting should be stubbed in tests")
    )
    pyplot_stub.tight_layout = lambda: None
    pyplot_stub.close = lambda fig: None
    matplotlib_stub.pyplot = pyplot_stub
    sys.modules["matplotlib"] = matplotlib_stub
    sys.modules["matplotlib.pyplot"] = pyplot_stub

MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts" / "example_inference.py"
MODULE_SPEC = importlib.util.spec_from_file_location("example_inference", MODULE_PATH)
assert MODULE_SPEC is not None
assert MODULE_SPEC.loader is not None
example_inference = importlib.util.module_from_spec(MODULE_SPEC)
MODULE_SPEC.loader.exec_module(example_inference)


def test_compute_expanded_rewards_per_frame_batches_prefix_samples(monkeypatch) -> None:
    frames = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
    captured_requests: list[list[dict]] = []

    def fake_post_evaluate_batch_npy(
        eval_server_url: str,
        samples: list[dict],
        timeout_s: float = 120.0,
        use_frame_steps: bool = False,
    ) -> dict:
        del eval_server_url, timeout_s
        captured_requests.append(samples)
        assert use_frame_steps is False
        assert len(samples) == 4
        for prefix_length, sample in enumerate(samples, start=1):
            np.testing.assert_array_equal(sample["trajectory"]["frames"], frames[:prefix_length])
            assert sample["trajectory"]["frames_shape"] == tuple(frames[:prefix_length].shape)
            assert sample["trajectory"]["metadata"]["subsequence_length"] == prefix_length

        return {
            "outputs_progress": {
                "progress_pred": [
                    [0.1],
                    [0.2, 0.25],
                    [0.3, 0.35, 0.4],
                    [0.4, 0.45, 0.5, 0.55],
                ]
            },
            "outputs_success": {
                "success_probs": [
                    [0.6],
                    [0.61, 0.62],
                    [0.63, 0.64, 0.65],
                    [0.66, 0.67, 0.68, 0.69],
                ]
            },
        }

    monkeypatch.setattr(example_inference, "post_evaluate_batch_npy", fake_post_evaluate_batch_npy)

    rewards, success_probs = example_inference.compute_expanded_rewards_per_frame(
        eval_server_url="http://localhost:8000",
        video_frames=frames,
        task="Stack blocks",
    )

    assert len(captured_requests) == 1
    np.testing.assert_allclose(rewards, np.array([0.1, 0.25, 0.4, 0.55], dtype=np.float32))
    np.testing.assert_allclose(success_probs, np.array([0.6, 0.62, 0.65, 0.69], dtype=np.float32))


def test_main_routes_expand_through_expanded_path(tmp_path: Path, monkeypatch) -> None:
    out_path = tmp_path / "expanded_rewards.npy"
    frames = np.zeros((3, 8, 8, 3), dtype=np.uint8)
    captured_kwargs: dict = {}

    class DummyFigure:
        def savefig(self, path: str, dpi: int = 200) -> None:
            del dpi
            Path(path).write_bytes(b"png")

    def fail_compute_rewards_per_frame(**kwargs):
        del kwargs
        raise AssertionError("compute_rewards_per_frame should not be used for --expand")

    def fake_compute_expanded_rewards_per_frame(**kwargs):
        captured_kwargs.update(kwargs)
        return (
            np.array([0.2, 0.4, 0.8], dtype=np.float32),
            np.array([0.1, 0.3, 0.7], dtype=np.float32),
        )

    monkeypatch.setattr(example_inference, "load_frames_input", lambda *args, **kwargs: frames)
    monkeypatch.setattr(example_inference, "compute_rewards_per_frame", fail_compute_rewards_per_frame)
    monkeypatch.setattr(
        example_inference,
        "compute_expanded_rewards_per_frame",
        fake_compute_expanded_rewards_per_frame,
        raising=False,
    )
    monkeypatch.setattr(
        example_inference,
        "create_combined_progress_success_plot",
        lambda **kwargs: DummyFigure(),
    )
    monkeypatch.setattr(example_inference.plt, "close", lambda fig: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "example_inference.py",
            "--video",
            str(tmp_path / "demo.npy"),
            "--task",
            "Stack blocks",
            "--expand",
            "--out",
            str(out_path),
        ],
    )

    example_inference.main()

    assert captured_kwargs["task"] == "Stack blocks"
    assert captured_kwargs["video_frames"] is frames
    np.testing.assert_allclose(np.load(out_path), np.array([0.2, 0.4, 0.8], dtype=np.float32))
    np.testing.assert_allclose(
        np.load(out_path.with_name(out_path.stem + "_success_probs.npy")),
        np.array([0.1, 0.3, 0.7], dtype=np.float32),
    )
    assert out_path.with_name(out_path.stem + "_progress_success.png").exists()


def test_main_rejects_expand_with_use_frame_steps(monkeypatch, capsys) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "example_inference.py",
            "--video",
            "demo.mp4",
            "--task",
            "Stack blocks",
            "--expand",
            "--use-frame-steps",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        example_inference.main()

    assert exc_info.value.code == 2
    assert "--expand cannot be used with --use-frame-steps" in capsys.readouterr().err
