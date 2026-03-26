"""OpenVLA-style synthetic policy backed by a learned linear checkpoint."""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np

from .base_policy import BasePolicy


class OpenVLA(BasePolicy):
    def load_model(self) -> None:
        path = self._resolve_checkpoint_path()
        if path.exists():
            with np.load(path, allow_pickle=False) as data:
                self.model = {key: data[key] for key in data.files}
        else:
            self.model = self._default_model()
            self.model["noise_scale"] = np.float32(0.045)

    def reset(self, instruction: str) -> None:
        self.instruction = instruction

    def act(self, observation: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        assert self.model is not None, "load_model() must be called first"
        x = self._state_vector(observation)
        weights = np.asarray(self.model["action_weights"], dtype=np.float32)
        bias = np.asarray(self.model["action_bias"], dtype=np.float32)
        conf_weights = np.asarray(self.model["confidence_weights"], dtype=np.float32)
        conf_bias = float(np.asarray(self.model["confidence_bias"]))
        noise_scale = float(np.asarray(self.model["noise_scale"]))
        num_mc_samples = int(self.config.get("num_mc_samples", 10))

        actions = []
        confidences = []
        for _ in range(max(num_mc_samples, 1)):
            latent = weights @ x + bias + self.rng.normal(0.0, noise_scale, size=7)
            action = np.tanh(latent).astype(np.float32)
            conf_logit = float(1.4 * (conf_weights @ x + conf_bias) + 0.55 - 0.18 * np.var(action) + self.rng.normal(0.0, noise_scale))
            confidence = float(np.clip(self._sigmoid(conf_logit), 1e-4, 1 - 1e-4))
            actions.append(action)
            confidences.append(confidence)

        actions_arr = np.stack(actions, axis=0)
        confidences_arr = np.asarray(confidences, dtype=np.float32)
        p = float(confidences_arr.mean())
        epsilon = 1e-6
        entropy = float(-p * np.log(p + epsilon) - (1 - p) * np.log(1 - p + epsilon))
        info = {
            "raw_confidence_samples": confidences_arr,
            "raw_action_samples": actions_arr,
            "raw_policy_confidence": p,
            "uncertainty_variance": float(actions_arr.var(axis=0).mean()),
            "uncertainty_entropy": entropy,
        }
        return actions_arr.mean(axis=0).astype(np.float32), info
