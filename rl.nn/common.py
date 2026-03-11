"""Common utilities for RL NN agents."""

import json

import matplotlib.pyplot as plt
import numpy as np


def _window_mean(values, window=100):
    arr = np.array(values, dtype=float)
    cumsum = np.cumsum(arr)
    means = np.zeros(len(arr))
    means[:window] = cumsum[:window] / np.arange(1, min(window, len(arr)) + 1)
    if len(arr) > window:
        means[window:] = (cumsum[window:] - cumsum[:-window]) / window
    return means


def plot_result(file_name: str) -> None:
    """Plots results from a JSON file."""
    with open(file_name, encoding="utf-8") as f:
        data = json.load(f)

    params = data.get("params", {})
    rewards = data["rewards"]
    epsilons = data.get("epsilons")
    has_epsilons = bool(epsilons)

    title = ", ".join(f"{k}={v}" for k, v in params.items())

    fig, ax1 = plt.subplots(figsize=(10, 5))
    fig.suptitle(title, fontsize=8)

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Reward")
    ax1.plot(rewards, alpha=0.4, label="reward")
    ax1.plot(_window_mean(rewards), label="mean")

    if has_epsilons:
        ax2 = ax1.twinx()
        ax2.set_ylabel("Epsilon")
        ax2.plot(epsilons, color="tab:red", alpha=0.5, label="epsilon")
        ax2.legend(fontsize=8, loc="upper right")

    ax1.legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    plt.show()
