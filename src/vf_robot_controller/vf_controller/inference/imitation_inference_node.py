#!/usr/bin/env python3
"""
imitation_inference_node.py — Phase 8 Python sidecar for IMITATION mode.

Subscribes to /vf/features (Float32MultiArray @ 20 Hz), runs the trained
imitation model, publishes /vf_controller/imitation_cmd_vel
(geometry_msgs/Twist) at the same rate.

The C++ ImitationVelocityProvider (loaded by VFController in IMITATION mode)
caches the latest message with a 200 ms staleness check and returns it as the
velocity command instead of running the MPPI optimizer.

Backend selection identical to metacritic_inference_node.py — onnxruntime
preferred, torch fallback, finally a zero-twist failsafe.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np

import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
from rclpy.node import Node
from rclpy.qos import QoSHistoryPolicy, QoSProfile, QoSReliabilityPolicy
from std_msgs.msg import Float32MultiArray


def _load_metadata(meta_path: str) -> dict:
    if not os.path.exists(meta_path):
        return {}
    try:
        with open(meta_path) as f:
            return json.load(f)
    except Exception:
        return {}


class ImitationInferenceNode(Node):
    def __init__(self) -> None:
        super().__init__("imitation_inference_node")

        self.declare_parameter("onnx_path", "")
        self.declare_parameter("pt_path", "")
        self.declare_parameter("norm_path", "")
        self.declare_parameter("metadata_path", "")
        # Default = /cmd_vel_nav: the sidecar drives Nav2's velocity_smoother
        # directly while VFController sits in PASSIVE mode. The legacy topic
        # "/vf_controller/imitation_cmd_vel" was for a C++ ImitationVelocityProvider
        # path that is no longer reachable from any active YAML; override
        # publish_topic if you wire that path back up.
        self.declare_parameter("publish_topic", "/cmd_vel_nav")
        self.declare_parameter("features_topic", "/vf/features")
        self.declare_parameter("expected_in_dim", 170)
        self.declare_parameter("vx_max", 0.30)
        self.declare_parameter("wz_max", 1.00)
        # Cold-start warmup. The trained model learns "frame 0 ≈ zero output"
        # because MPPI itself commanded ~0 at training-frame 0 (then ramped via
        # internal trajectory state). The imitation policy is Markov, so it has
        # no accumulator to escape the fixed point. We override the model with
        # a small forward velocity for `warmup_seconds` after each new global
        # plan arrives — that's enough to populate robot_state with motion and
        # let the model take over for the rest of the episode.
        self.declare_parameter("plan_topic", "/plan")
        self.declare_parameter("warmup_seconds", 1.0)
        # 0.15 m/s lands in the velocity regime where the trained model wants
        # to accelerate (pred vx ≈ +0.14 at |v|∈[0.10, 0.20]). Lower kicks
        # (0.08) leave the robot in the model's near-zero bucket and the
        # robot decelerates back to rest after warmup ends.
        self.declare_parameter("warmup_vx", 0.15)
        self.declare_parameter("warmup_wz", 0.0)
        # Speed threshold below which the robot is considered "at rest" for the
        # purpose of arming a warmup kick. NavFn replans at ~1 Hz, so without a
        # rest gate the warmup re-arms every cycle and the robot drives forever
        # at warmup_vx instead of handing over to the model.
        self.declare_parameter("warmup_rest_speed", 0.05)

        self.onnx_path = self.get_parameter("onnx_path").value
        self.pt_path = self.get_parameter("pt_path").value
        self.norm_path = self.get_parameter("norm_path").value
        self.metadata_path = self.get_parameter("metadata_path").value
        self.expected_in_dim = int(self.get_parameter("expected_in_dim").value)
        self.vx_max = float(self.get_parameter("vx_max").value)
        self.wz_max = float(self.get_parameter("wz_max").value)
        self.warmup_seconds = float(self.get_parameter("warmup_seconds").value)
        self.warmup_vx = float(self.get_parameter("warmup_vx").value)
        self.warmup_wz = float(self.get_parameter("warmup_wz").value)
        self.warmup_rest_speed = float(self.get_parameter("warmup_rest_speed").value)
        self._warmup_until_ns: int = 0
        self._latest_speed: float = 0.0  # |v| from most recent feature msg

        self._backend = "zero"
        self._sess = None
        self._torch_model = None
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._warned_dim = False
        self._warned_infer = False

        meta = _load_metadata(self.metadata_path) if self.metadata_path else {}
        if meta and "in_dim" in meta:
            self.expected_in_dim = int(meta["in_dim"])

        self._setup_norm()
        self._setup_backend()

        sub_qos = QoSProfile(depth=10)
        sub_qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        sub_qos.history = QoSHistoryPolicy.KEEP_LAST

        self.pub = self.create_publisher(
            Twist, self.get_parameter("publish_topic").value, 10
        )
        self.sub = self.create_subscription(
            Float32MultiArray,
            self.get_parameter("features_topic").value,
            self._on_features,
            sub_qos,
        )
        self.plan_sub = self.create_subscription(
            Path, self.get_parameter("plan_topic").value, self._on_plan, 10
        )

        self.get_logger().info(
            f"imitation_inference_node: backend={self._backend} "
            f"in_dim={self.expected_in_dim} "
            f"onnx={self.onnx_path or '(none)'} pt={self.pt_path or '(none)'}"
        )
        if self._backend == "zero":
            self.get_logger().error(
                "IMITATION SIDECAR IS IN ZERO-TWIST FAILSAFE. Robot will not "
                "move. Neither ONNX nor PT loaded — check that the model file "
                f"exists at onnx_path={self.onnx_path!r} pt_path={self.pt_path!r}. "
                "Most common cause: vf_robot_controller models/ not installed "
                "into share/ (rebuild after adding install(DIRECTORY models ...))."
            )

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
        if self.onnx_path and os.path.exists(self.onnx_path):
            try:
                import onnxruntime as ort  # type: ignore

                self._sess = ort.InferenceSession(
                    self.onnx_path, providers=["CPUExecutionProvider"]
                )
                # Read actual input dim from the model rather than relying on
                # the expected_in_dim parameter (avoids silent zero-twist when
                # the two disagree, e.g. channels_v1 model on a channels_v3 node).
                model_in_dim = int(self._sess.get_inputs()[0].shape[1])
                if model_in_dim != self.expected_in_dim:
                    self.get_logger().warn(
                        f"expected_in_dim={self.expected_in_dim} overridden by "
                        f"ONNX model input shape: {model_in_dim}"
                    )
                    self.expected_in_dim = model_in_dim
                self._backend = "onnx"
                return
            except Exception as e:
                self.get_logger().warn(f"onnxruntime load failed ({e}); trying torch.")

        if self.pt_path and os.path.exists(self.pt_path):
            try:
                import torch
                from vf_controller.training.models.imitation_net import ImitationNet

                state = torch.load(self.pt_path, map_location="cpu", weights_only=True)
                # Infer in_dim from the first layer weight shape.
                pt_in_dim = int(next(iter(state.values())).shape[-1])
                if pt_in_dim != self.expected_in_dim:
                    self.get_logger().warn(
                        f"expected_in_dim={self.expected_in_dim} overridden by "
                        f"checkpoint first-layer shape: {pt_in_dim}"
                    )
                    self.expected_in_dim = pt_in_dim
                model = ImitationNet(
                    in_dim=self.expected_in_dim, vx_max=self.vx_max, wz_max=self.wz_max
                )
                model.load_state_dict(state, strict=True)
                model.eval()
                self._torch_model = model
                self._backend = "torch"
                return
            except Exception as e:
                self.get_logger().warn(
                    f"torch load failed ({e}); falling back to zero-twist."
                )

        self._backend = "zero"

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        if self._mean is None or self._std is None or self._mean.shape != x.shape:
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
        return np.zeros(2, dtype=np.float32)

    def _on_plan(self, msg: Path) -> None:
        # A new global plan arrived. Only arm the warmup if (a) we have features
        # to base the gate on, (b) the robot is at rest, and (c) we are not
        # already inside a warmup window. NavFn replans at ~1 Hz, so without
        # the rest gate the warmup would re-arm every cycle and the robot
        # would drive forever at warmup_vx.
        if self.warmup_seconds <= 0.0 or len(msg.poses) == 0:
            return
        now_ns = self.get_clock().now().nanoseconds
        if now_ns < self._warmup_until_ns:
            return  # already warming up
        if self._latest_speed >= self.warmup_rest_speed:
            return  # robot is already moving
        self._warmup_until_ns = now_ns + int(self.warmup_seconds * 1e9)
        self.get_logger().info(
            f"warmup armed: speed={self._latest_speed:.3f} m/s < "
            f"{self.warmup_rest_speed:.3f}; forcing vx={self.warmup_vx:+.3f} "
            f"wz={self.warmup_wz:+.3f} for {self.warmup_seconds:.2f}s."
        )

    def _on_features(self, msg: Float32MultiArray) -> None:
        # Warmup override takes precedence over model output. The model has a
        # zero-fixed-point at episode start; this small forward kick gets the
        # robot moving, populates robot_state with motion, then the model
        # takes over once the warmup window closes.
        now_ns = self.get_clock().now().nanoseconds
        if now_ns < self._warmup_until_ns:
            cmd = (
                float(np.clip(self.warmup_vx, -self.vx_max, self.vx_max)),
                float(np.clip(self.warmup_wz, -self.wz_max, self.wz_max)),
            )
            twist = Twist()
            twist.linear.x = cmd[0]
            twist.angular.z = cmd[1]
            self.pub.publish(twist)
            return

        x = np.asarray(msg.data, dtype=np.float32)
        if x.size != self.expected_in_dim:
            if not self._warned_dim:
                self.get_logger().warn(
                    f"feature dim {x.size} != expected {self.expected_in_dim}; "
                    f"publishing zero-twist."
                )
                self._warned_dim = True
            cmd = (0.0, 0.0)
        else:
            # robot_state layout (first 9 dims): [vx, vy, wz, sinθ, cosθ, |v|, ax, ay, αz]
            # Track |v| (raw, pre-normalisation) so _on_plan can gate the warmup.
            self._latest_speed = float(abs(x[5]))
            x_n = self._normalize(x)
            try:
                pred = self._infer(x_n)
                vx = float(np.clip(pred[0], -self.vx_max, self.vx_max))
                wz = float(np.clip(pred[1], -self.wz_max, self.wz_max))
                cmd = (vx, wz)
            except Exception as e:
                if not self._warned_infer:
                    self.get_logger().warn(f"inference exception: {e}")
                    self._warned_infer = True
                cmd = (0.0, 0.0)

        twist = Twist()
        twist.linear.x = cmd[0]
        twist.angular.z = cmd[1]
        self.pub.publish(twist)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ImitationInferenceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
