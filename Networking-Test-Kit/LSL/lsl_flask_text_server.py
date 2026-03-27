"""Flask server that bridges an LSL EEG stream to a /text HTTP endpoint.

The server pulls samples from the first LSL EEG stream it finds, computes a
plain-text neural feedback summary, and exposes it at GET /text.  All
generated text is strictly ASCII so that it can be saved or embedded in
Python source files without triggering a SyntaxError from non-ASCII
characters such as emoji.

Install dependencies:
    pip install flask pylsl

Run:
    python lsl_flask_text_server.py [--host HOST] [--port PORT]
             [--stream-type STREAM_TYPE] [--window-seconds WINDOW_SECONDS]
             [--fallback-rate FALLBACK_RATE]
"""

import argparse
import math
import threading
import time

from flask import Flask, jsonify
from pylsl import StreamInlet, resolve_byprop

# ---------------------------------------------------------------------------
# Default configuration
# ---------------------------------------------------------------------------
DEFAULT_STREAM_TYPE = "EEG"
DEFAULT_WINDOW_SECONDS = 2.0
DEFAULT_FALLBACK_RATE = 250.0  # Hz -- used when the stream does not report its rate
DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5000

# ---------------------------------------------------------------------------
# Neural feedback text generation (ASCII-only output)
# ---------------------------------------------------------------------------


def _mean(values):
    return sum(values) / len(values)


def _stddev(values):
    m = _mean(values)
    variance = sum((v - m) ** 2 for v in values) / len(values)
    return math.sqrt(variance)


def _mean_abs_delta(values):
    if len(values) < 2:
        return 0.0
    deltas = [abs(b - a) for a, b in zip(values, values[1:])]
    return _mean(deltas)


def _infer_state(avg_stddev, avg_peak_to_peak, avg_abs_delta):
    """Return a (state_label, interpretation) tuple based on signal metrics.

    All strings returned here are pure ASCII so they are safe to embed in
    Python source files or transmit over HTTP without encoding issues.
    """
    score = avg_stddev + 0.5 * avg_peak_to_peak + avg_abs_delta
    if score < 20:
        return (
            "steady / low-activation",
            "Signal changes are small, which typically corresponds to a calm or stable feedback window.",
        )
    if score < 60:
        return (
            "balanced / moderate-activation",
            "Signal energy is present without large swings, suggesting a moderately engaged feedback window.",
        )
    return (
        "active / high-variation",
        "Signal energy and variation are elevated, suggesting an active or strongly changing feedback window.",
    )


def build_text_summary(samples, nominal_rate, fallback_rate):
    """Convert a list of multi-channel samples into a plain-text summary.

    Parameters
    ----------
    samples:
        List of sample lists, each inner list being one sample across all
        channels.
    nominal_rate:
        Nominal sampling rate reported by the LSL stream.  When this is
        zero or negative the *fallback_rate* is used instead.
    fallback_rate:
        Rate (Hz) to use when the stream does not report a valid rate.

    Returns
    -------
    str
        A multi-line, ASCII-only text summary.
    """
    if not samples:
        return "No samples were received from the LSL stream."

    n_channels = len(samples[0])
    if n_channels == 0:
        return "The LSL stream reported zero channels."

    effective_rate = nominal_rate if nominal_rate > 0 else fallback_rate
    window_seconds = len(samples) / effective_rate

    channel_stats = []
    for ch in range(n_channels):
        ch_data = [sample[ch] for sample in samples]
        if len(ch_data) < 2:
            continue
        channel_stats.append(
            {
                "channel": ch,
                "mean": _mean(ch_data),
                "stddev": _stddev(ch_data),
                "peak_to_peak": max(ch_data) - min(ch_data),
                "mean_abs_delta": _mean_abs_delta(ch_data),
            }
        )

    if not channel_stats:
        return "Not enough samples to compute per-channel statistics."

    avg_stddev = _mean([s["stddev"] for s in channel_stats])
    avg_peak_to_peak = _mean([s["peak_to_peak"] for s in channel_stats])
    avg_abs_delta = _mean([s["mean_abs_delta"] for s in channel_stats])

    state_label, interpretation = _infer_state(avg_stddev, avg_peak_to_peak, avg_abs_delta)

    lines = [
        "Neural feedback text summary",
        f"Window length: {window_seconds:.2f} seconds ({len(samples)} samples)",
        f"Effective sampling rate: {effective_rate:.1f} Hz",
        f"Number of channels: {n_channels}",
        f"Detected state: {state_label}",
        f"Interpretation: {interpretation}",
        "Channel highlights:",
    ]
    for stats in channel_stats:
        lines.append(
            f"  - Channel {stats['channel']}: "
            f"mean={stats['mean']:.3f}, "
            f"stddev={stats['stddev']:.3f}, "
            f"peak_to_peak={stats['peak_to_peak']:.3f}, "
            f"mean_abs_delta={stats['mean_abs_delta']:.3f}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Background LSL reader
# ---------------------------------------------------------------------------


class LSLReader(threading.Thread):
    """Background thread that continuously pulls samples from an LSL inlet."""

    def __init__(self, stream_type, window_seconds, fallback_rate):
        super().__init__(daemon=True)
        self._stream_type = stream_type
        self._window_seconds = window_seconds
        self._fallback_rate = fallback_rate
        self._lock = threading.Lock()
        self._samples = []
        self._nominal_rate = 0.0
        self._status = "Waiting for LSL stream..."

    # ------------------------------------------------------------------
    # Public interface (thread-safe)
    # ------------------------------------------------------------------

    def get_snapshot(self):
        """Return a copy of the current sample window and metadata."""
        with self._lock:
            return list(self._samples), self._nominal_rate, self._status

    @property
    def fallback_rate(self):
        """The fallback sampling rate (Hz) used when the stream rate is unavailable."""
        return self._fallback_rate

    # ------------------------------------------------------------------
    # Thread body
    # ------------------------------------------------------------------

    def run(self):
        while True:
            self._status = f"Resolving LSL stream of type '{self._stream_type}'..."
            streams = resolve_byprop("type", self._stream_type, timeout=5)
            if not streams:
                self._status = f"No LSL stream of type '{self._stream_type}' found. Retrying..."
                time.sleep(2)
                continue

            inlet = StreamInlet(streams[0])
            info = inlet.info()
            nominal_rate = info.nominal_srate()
            effective_rate = nominal_rate if nominal_rate > 0 else self._fallback_rate
            max_samples = int(self._window_seconds * effective_rate)

            with self._lock:
                self._nominal_rate = nominal_rate
                self._status = (
                    f"Connected. Nominal rate: "
                    f"{nominal_rate if nominal_rate > 0 else self._fallback_rate:.1f} Hz"
                )

            buffer = []
            try:
                while True:
                    sample, _ = inlet.pull_sample(timeout=2.0)
                    if sample is None:
                        continue
                    buffer.append(sample)
                    if len(buffer) > max_samples:
                        buffer = buffer[-max_samples:]
                    with self._lock:
                        self._samples = list(buffer)
            except Exception as exc:  # pylint: disable=broad-except
                with self._lock:
                    self._status = f"Stream disconnected: {exc}. Reconnecting..."
                time.sleep(1)


# ---------------------------------------------------------------------------
# Flask application
# ---------------------------------------------------------------------------


def create_app(reader):
    app = Flask(__name__)

    @app.route("/text")
    def text_endpoint():
        samples, nominal_rate, status = reader.get_snapshot()
        summary = build_text_summary(samples, nominal_rate, reader.fallback_rate)
        return jsonify(
            {
                "status": status,
                "summary": summary,
                "sample_count": len(samples),
            }
        )

    @app.route("/health")
    def health_endpoint():
        _, _, status = reader.get_snapshot()
        return jsonify({"status": status})

    return app


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Serve LSL EEG stream data as plain-text neural feedback via HTTP."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host to bind to")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to listen on")
    parser.add_argument(
        "--stream-type",
        default=DEFAULT_STREAM_TYPE,
        help="LSL stream type to subscribe to",
    )
    parser.add_argument(
        "--window-seconds",
        type=float,
        default=DEFAULT_WINDOW_SECONDS,
        help="Rolling window of samples to include in each summary",
    )
    parser.add_argument(
        "--fallback-rate",
        type=float,
        default=DEFAULT_FALLBACK_RATE,
        help=(
            "Sampling rate (Hz) to use when the LSL stream does not report a"
            " valid nominal rate"
        ),
    )
    args = parser.parse_args()

    reader = LSLReader(
        stream_type=args.stream_type,
        window_seconds=args.window_seconds,
        fallback_rate=args.fallback_rate,
    )
    reader.start()

    app = create_app(reader)
    print(
        f"Starting LSL Flask text server on http://{args.host}:{args.port}/text"
    )
    app.run(host=args.host, port=args.port)


if __name__ == "__main__":
    main()
