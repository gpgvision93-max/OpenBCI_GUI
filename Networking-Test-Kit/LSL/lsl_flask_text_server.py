"""LSL + Flask bridge for streaming EEG text summaries to a phone.

Pipeline:
OpenBCI GUI (LSL) -> PC (this Flask server) -> iPhone (HTTP) -> text output.
"""

import argparse
import math
import threading
import time
from collections import deque
from dataclasses import dataclass

from flask import Flask, jsonify
from pylsl import StreamInlet, resolve_byprop


@dataclass
class ChannelFeatures:
    channel_index: int
    sample_count: int
    mean: float
    stddev: float
    peak_to_peak: float
    mean_abs_delta: float


class NeuralFeedbackTextTransformer:
    def __init__(self, max_channels=4):
        self.max_channels = max_channels

    def transform(self, samples, sample_rate, channel_labels=None):
        if not samples:
            return "Waiting for EEG samples..."

        channel_count = len(samples[0])
        if channel_count == 0:
            return "No EEG channels were found in the LSL stream."

        per_channel = list(zip(*samples))
        channel_features = []
        for channel_index, channel_data in enumerate(
            per_channel[: self.max_channels]
        ):
            values = [float(value) for value in channel_data]
            if len(values) < 2:
                continue
            channel_features.append(
                ChannelFeatures(
                    channel_index=channel_index,
                    sample_count=len(values),
                    mean=self._mean(values),
                    stddev=self._stddev(values),
                    peak_to_peak=max(values) - min(values),
                    mean_abs_delta=self._mean_abs_delta(values),
                )
            )

        if not channel_features:
            return "The EEG stream did not contain enough samples to generate a text summary."

        sample_count = channel_features[0].sample_count
        window_seconds = sample_count / sample_rate if sample_rate else 0.0

        average_stddev = self._mean([item.stddev for item in channel_features])
        average_peak_to_peak = self._mean(
            [item.peak_to_peak for item in channel_features]
        )
        average_abs_delta = self._mean(
            [item.mean_abs_delta for item in channel_features]
        )

        state_label, interpretation = self._infer_state(
            average_stddev, average_peak_to_peak, average_abs_delta
        )

        lines = [
            "Neural feedback text summary",
            f"Window length: {window_seconds:.2f} seconds",
            f"Samples per channel: {sample_count}",
            f"Detected state: {state_label}",
            f"Interpretation: {interpretation}",
            "Channel highlights:",
        ]

        for feature in channel_features:
            label = (
                channel_labels[feature.channel_index]
                if channel_labels and feature.channel_index < len(channel_labels)
                else f"EEG {feature.channel_index}"
            )
            lines.append(
                "  - "
                f"{label}: mean={feature.mean:.3f}, "
                f"stddev={feature.stddev:.3f}, "
                f"peak_to_peak={feature.peak_to_peak:.3f}, "
                f"mean_abs_delta={feature.mean_abs_delta:.3f}"
            )

        return "\n".join(lines)

    def _infer_state(self, average_stddev, average_peak_to_peak, average_abs_delta):
        activation_score = average_stddev + (0.5 * average_peak_to_peak) + average_abs_delta

        if activation_score < 20:
            return (
                "steady / low-activation",
                "Signal changes are relatively small, which usually corresponds to a calm or stable feedback window.",
            )
        if activation_score < 60:
            return (
                "balanced / moderate-activation",
                "Signal energy is present without large swings, suggesting a moderately engaged feedback window.",
            )

        return (
            "active / high-variation",
            "Signal energy and short-term variation are elevated, suggesting an active or strongly changing feedback window.",
        )

    def _mean(self, values):
        return sum(values) / len(values)

    def _stddev(self, values):
        mean_value = self._mean(values)
        variance = sum((value - mean_value) ** 2 for value in values) / len(values)
        return math.sqrt(variance)

    def _mean_abs_delta(self, values):
        deltas = [abs(current - previous) for previous, current in zip(values, values[1:])]
        return self._mean(deltas)


class LslTextServer:
    def __init__(
        self,
        inlet,
        sample_rate,
        window_seconds,
        update_interval,
        max_channels,
        channel_labels,
    ):
        self.inlet = inlet
        self.sample_rate = sample_rate
        self.window_samples = max(1, int(window_seconds * sample_rate))
        self.update_interval = update_interval
        self.channel_labels = channel_labels
        self.transformer = NeuralFeedbackTextTransformer(max_channels=max_channels)
        self.buffer = deque(maxlen=self.window_samples or None)
        self.latest_summary = "Waiting for EEG samples..."
        self.last_update = 0.0
        self._stop = threading.Event()

    def start(self):
        thread = threading.Thread(target=self._collect_loop, daemon=True)
        thread.start()

    def stop(self):
        self._stop.set()

    def summary(self):
        return self.latest_summary

    def _collect_loop(self):
        while not self._stop.is_set():
            chunk, _timestamps = self.inlet.pull_chunk(timeout=1.0)
            if chunk:
                for sample in chunk:
                    if not isinstance(sample, list):
                        sample = list(sample)
                    self.buffer.append(sample)

            now = time.time()
            if (
                self.buffer
                and (now - self.last_update) >= self.update_interval
                and (not self.window_samples or len(self.buffer) >= self.window_samples)
            ):
                samples = list(self.buffer)
                self.latest_summary = self.transformer.transform(
                    samples, self.sample_rate, channel_labels=self.channel_labels
                )
                self.last_update = now


def _resolve_stream(stream_type, stream_name, timeout):
    if stream_name:
        streams = resolve_byprop("name", stream_name, timeout=timeout)
    else:
        streams = resolve_byprop("type", stream_type, timeout=timeout)

    if not streams:
        raise SystemExit(
            f"No LSL stream found for type='{stream_type}' name='{stream_name}'."
        )

    return streams[0]


def create_app(server):
    app = Flask(__name__)

    @app.after_request
    def allow_phone_access(response):
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "GET"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
        return response

    @app.route("/health")
    def health():
        return jsonify({"status": "ok", "timestamp": time.time()})

    @app.route("/text")
    def text_summary():
        return jsonify(
            {
                "summary": server.summary(),
                "timestamp": time.time(),
                "sample_rate": server.sample_rate,
                "window_samples": server.window_samples,
                "buffered_samples": len(server.buffer),
            }
        )

    return app


def main():
    parser = argparse.ArgumentParser(
        description="Bridge LSL EEG data to a Flask endpoint for iPhone text output."
    )
    parser.add_argument("--stream-type", default="EEG", help="LSL stream type.")
    parser.add_argument("--stream-name", default="", help="LSL stream name.")
    parser.add_argument("--timeout", type=float, default=10.0, help="LSL resolve timeout.")
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=2.0,
        help="EEG window duration for each text summary.",
    )
    parser.add_argument(
        "--update-interval",
        type=float,
        default=1.0,
        help="Seconds between summary refreshes.",
    )
    parser.add_argument("--max-channels", type=int, default=4)
    parser.add_argument(
        "--channel-labels",
        default="",
        help="Comma-separated channel labels (optional).",
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--fallback-sample-rate",
        type=float,
        default=250.0,
        help="Sample rate to assume when LSL does not report one.",
    )
    args = parser.parse_args()

    stream = _resolve_stream(args.stream_type, args.stream_name, args.timeout)
    inlet = StreamInlet(stream)
    info = inlet.info()
    reported_rate = info.nominal_srate()
    sample_rate = reported_rate if reported_rate else args.fallback_sample_rate
    channel_labels = [label.strip() for label in args.channel_labels.split(",") if label.strip()]

    server = LslTextServer(
        inlet=inlet,
        sample_rate=sample_rate,
        window_seconds=args.window_seconds,
        update_interval=args.update_interval,
        max_channels=args.max_channels,
        channel_labels=channel_labels,
    )
    server.start()

    app = create_app(server)
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
