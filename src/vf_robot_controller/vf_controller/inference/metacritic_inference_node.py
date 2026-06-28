#!/usr/bin/env python3
"""
metacritic_inference_node.py — Phase 8 Python sidecar for INFERENCE mode.

Subscribes to /vf/features (Float32MultiArray @ 20 Hz), runs the trained
meta-critic model, publishes /vf_controller/meta_weights (Float32MultiArray)
at the same rate. The C++ TopicWeightProvider in the controller reads this
topic and applies the weights to the upstream MPPI critics each cycle.

Why a Python sidecar even though OnnxWeightProvider exists in C++:
  * The package's runtime onnxruntime C++ library is not guaranteed to be
    installed on every machine (the design notes: "If onnxruntime not findable on
    this system, propose the smallest workaround"). The Python torch path
    is always available because the dl conda env ships it. Running the
    network in Python and shipping weights over a topic is the
    development-mode and fall-back pattern — it gives us a correct
    INFERENCE row in the benchmark even if onnxruntime-c++ is missing.
  * A C++ OnnxWeightProvider is also provided (in-process, no topic round
    trip) for production deployments where a small dependency is OK.

Inference backend selection (in priority order):
  1. onnxruntime python   if available (fast, matches the C++ path)
  2. torch                always available in the dl env (uses .pt + the
                          model class definition)
  3. uniform fall-back    if neither works — log warning, publish ones.
"""
from __future__ import annotations

import json
import os
import time
from typing import List, Optional

import numpy as np

import rclpy
from rclpy.node import Node
from rclpy.qos import (QoSDurabilityPolicy, QoSHistoryPolicy, QoSProfile,
                       QoSReliabilityPolicy)
from std_msgs.msg import Float32MultiArray, MultiArrayDimension


def _load_metadata(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


def _build_net_from_state(state: dict, n_critics: int):
    """Construct an InferenceNet whose architecture matches the state dict.

    InferenceNet has configurable per_channel_hidden and fusion_hidden that
    may differ from the defaults — e.g. oracle model uses per_h=48 while
    raw_critics uses per_h=32.  Read the shapes directly from the checkpoint
    so we never silently load into a mismatched skeleton.
    """
    from vf_controller.training.models.inference_net import InferenceNet
    # per-channel hidden: output dim of each channel MLP's first linear
    per_h = int(state["frontend.per_channel.0.0.weight"].shape[0])
    in_dim = int(state["frontend.per_channel.0.0.weight"].shape[1])
    # fusion hidden: collect output dims of all 2-D weight tensors in fusion,
    # sorted by layer index.  _mlp appends a final Linear(out_dim, out_dim)
    # where out_dim = fusion_hidden[-1], so drop the last entry to recover
    # the hidden tuple.  Example: [256,128,64,64] -> (256,128,64).
    fusion_linear_dims = [
        int(v.shape[0])
        for _, v in sorted(
            ((int(k.split(".")[2]), v)
             for k, v in state.items()
             if k.startswith("frontend.fusion.")
             and k.endswith(".weight")
             and v.dim() == 2),
            key=lambda t: t[0],
        )
    ]
    fusion_hidden = tuple(fusion_linear_dims[:-1]) if len(fusion_linear_dims) > 1 \
        else (fusion_linear_dims[0],) if fusion_linear_dims else (64,)
    return InferenceNet(
        in_dim=in_dim,
        n_critics=n_critics,
        per_channel_hidden=per_h,
        fusion_hidden=fusion_hidden,
    )


class MetaCriticInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("metacritic_inference_node")

        self.declare_parameter("onnx_path", "")
        self.declare_parameter("pt_path", "")
        self.declare_parameter("norm_path", "")
        self.declare_parameter("metadata_path", "")
        self.declare_parameter("publish_topic", "/vf_controller/meta_weights")
        self.declare_parameter("features_topic", "/vf/features")
        self.declare_parameter("n_critics", 11)
        self.declare_parameter("expected_in_dim", 170)
        self.declare_parameter("publish_on_features", True)

        self.onnx_path = self.get_parameter("onnx_path").value
        self.pt_path = self.get_parameter("pt_path").value
        self.norm_path = self.get_parameter("norm_path").value
        self.metadata_path = self.get_parameter("metadata_path").value
        self.n_critics = int(self.get_parameter("n_critics").value)
        self.expected_in_dim = int(self.get_parameter("expected_in_dim").value)

        # ── Set up backend ────────────────────────────────────────────────
        self._backend = "uniform"
        self._sess = None
        self._torch_model = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._warned_dim = False
        self._warned_infer = False

        meta = _load_metadata(self.metadata_path) if self.metadata_path else {}
        if meta and "in_dim" in meta:
            self.expected_in_dim = int(meta["in_dim"])
        if meta and "target_dim" in meta:
            self.n_critics = int(meta["target_dim"])

        self._setup_norm()
        self._setup_backend()

        # ── Pub/Sub ───────────────────────────────────────────────────────
        sub_qos = QoSProfile(depth=10)
        sub_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        sub_qos.history = QoSHistoryPolicy.KEEP_LAST

        pub_qos = QoSProfile(depth=10)

        self.pub = self.create_publisher(
            Float32MultiArray,
            self.get_parameter("publish_topic").value, pub_qos)
        self.sub = self.create_subscription(
            Float32MultiArray,
            self.get_parameter("features_topic").value,
            self._on_features, sub_qos)

        self.get_logger().info(
            f"metacritic_inference_node: backend={self._backend} "
            f"in_dim={self.expected_in_dim} K={self.n_critics} "
            f"onnx={self.onnx_path or '(none)'} pt={self.pt_path or '(none)'}")
        if self._backend == "uniform":
            self.get_logger().error(
                "METACRITIC SIDECAR IS PUBLISHING UNIFORM WEIGHTS (1.0 for all "
                f"{self.n_critics} critics). Recorded runs will NOT reflect the "
                "trained model — they are equivalent to a constant-weight "
                "baseline. Neither ONNX nor PT loaded — check onnx_path="
                f"{self.onnx_path!r} pt_path={self.pt_path!r}. Most common "
                "cause: vf_robot_controller models/ not installed into share/."
            )

    # ----------------------------------------------------------------- setup
    def _setup_norm(self) -> None:
        if not self.norm_path or not os.path.exists(self.norm_path):
            return
        try:
            with open(self.norm_path) as f:
                d = json.load(f)
            self._mean = np.asarray(d["mean"], dtype=np.float32)
            self._std = np.asarray(d["std"], dtype=np.float32)
        except Exception as e:
            self.get_logger().warn(f"could not load norm: {e}")

    def _setup_backend(self) -> None:
        # 1) onnxruntime
        if self.onnx_path and os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort  # type: ignore
                self._sess = ort.InferenceSession(
                    self.onnx_path, providers=["CPUExecutionProvider"])
                model_in_dim = int(self._sess.get_inputs()[0].shape[1])
                if model_in_dim != self.expected_in_dim:
                    self.get_logger().warn(
                        f"expected_in_dim={self.expected_in_dim} overridden by "
                        f"ONNX model input shape: {model_in_dim}")
                    self.expected_in_dim = model_in_dim
                self._backend = "onnx"
                return
            except Exception as e:
                self.get_logger().warn(
                    f"onnxruntime load failed ({e}); falling back to torch.")

        # 2) torch
        if self.pt_path and os.path.exists(self.pt_path):
            try:
                import torch
                state = torch.load(self.pt_path, map_location="cpu",
                                   weights_only=True)
                model = _build_net_from_state(state, self.n_critics)
                model.load_state_dict(state)
                model.eval()
                self._torch_model = model
                self._backend = "torch"
                return
            except Exception as e:
                self.get_logger().warn(
                    f"torch load failed ({e}); falling back to uniform.")

        # 3) uniform
        self._backend = "uniform"

    # ---------------------------------------------------------------- runtime
    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None:
            return x
        if self._mean.shape != x.shape:
            return x
        out = (x - self._mean) / self._std
        return np.where(np.isfinite(out), out, 0.0).astype(np.float32)

    def _infer(self, x: np.ndarray) -> np.ndarray:
        if self._backend == "onnx" and self._sess is not None:
            in_name = self._sess.get_inputs()[0].name
            out = self._sess.run(None, {in_name: x[None].astype(np.float32)})[0]
            return np.asarray(out[0], dtype=np.float32)
        if self._backend == "torch" and self._torch_model is not None:
            import torch
            with torch.no_grad():
                t = torch.from_numpy(x[None].astype(np.float32))
                w = self._torch_model(t)[0].cpu().numpy()
            return w.astype(np.float32)
        return np.ones(self.n_critics, dtype=np.float32)

    def _on_features(self, msg: Float32MultiArray) -> None:
        x = np.asarray(msg.data, dtype=np.float32)
        if x.size != self.expected_in_dim:
            if not self._warned_dim:
                self.get_logger().warn(
                    f"feature dim {x.size} != expected {self.expected_in_dim}; "
                    f"publishing uniform.")
                self._warned_dim = True
            w = np.ones(self.n_critics, dtype=np.float32)
        else:
            x_n = self._normalize(x)
            try:
                w = self._infer(x_n)
            except Exception as e:
                if not self._warned_infer:
                    self.get_logger().warn(f"inference exception: {e}")
                    self._warned_infer = True
                w = np.ones(self.n_critics, dtype=np.float32)

        out = Float32MultiArray()
        out.layout.dim.append(MultiArrayDimension(label="critic", size=int(w.size),
                                                  stride=int(w.size)))
        out.data = [float(v) for v in w.tolist()]
        self.pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MetaCriticInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
