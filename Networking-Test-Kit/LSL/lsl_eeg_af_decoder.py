"""Real-time EEG → Afrikaans decoder using a sliding-window LSL inlet and a PyTorch transformer model."""
import np
import torch
from pylsl import StreamInlet, resolve_stream
from collections import deque
import time

# -----------------------------
# CONFIGURATION
# -----------------------------
CHANNELS = 8            # EEG channels
SAMPLING_RATE = 250     # Hz
WINDOW_SIZE = 250       # 1 second
STEP_SIZE = 50          # Sliding step
MODEL_PATH = "eeg_transformer.pt"

# Simple English → Afrikaans mapping for demo purposes
EN_TO_AF = {
    "LEFT": "LINKER",
    "RIGHT": "REGTER",
    "UP": "OP",
    "DOWN": "AF",
    "HELLO": "HALLO",
    "BYE": "TOTSIENS"
}

# -----------------------------
# CONNECT TO EEG STREAM
# -----------------------------
print("Resolving EEG stream...")
streams = resolve_stream('type', 'EEG')
inlet = StreamInlet(streams[0])
print("Connected to EEG stream.")

# Validate channel count against the stream metadata
stream_info = inlet.info()
stream_channels = stream_info.channel_count()
if stream_channels != CHANNELS:
    print(f"Warning: stream has {stream_channels} channels but CHANNELS={CHANNELS}. Updating CHANNELS.")
    CHANNELS = stream_channels

# -----------------------------
# LOAD TRANSFORMER MODEL
# -----------------------------
print("Loading transformer model...")
device = torch.device('cpu')  # change to 'cuda' if GPU available
model = torch.load(MODEL_PATH, map_location=device, weights_only=False)  # trusted source required
model.eval()
print("Model loaded.")

# -----------------------------
# REAL-TIME BUFFER
# -----------------------------
buffer = deque(maxlen=WINDOW_SIZE)
print("Starting real-time EEG → Afrikaans decoding...")

try:
    while True:
        sample, _ = inlet.pull_sample(timeout=1.0)
        if sample is None:
            continue

        buffer.append(sample)

        # Predict only when buffer full
        if len(buffer) == WINDOW_SIZE:
            segment = np.array(buffer)
            # Normalize per channel
            segment = (segment - np.mean(segment, axis=0)) / (np.std(segment, axis=0) + 1e-6)
            segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0)  # [1, TIME, CHANNELS]

            # Transformer prediction
            with torch.no_grad():
                logits = model(segment_tensor)
                predicted_class = torch.argmax(logits, dim=-1).item()

            # Map to Afrikaans
            # Note: model must expose a 'labels' list attribute (e.g. model.labels = ["LEFT", "RIGHT", ...])
            if hasattr(model, 'labels') and 0 <= predicted_class < len(model.labels):
                predicted_label_eng = model.labels[predicted_class]
            else:
                predicted_label_eng = "UNKNOWN"
            af_label = EN_TO_AF.get(predicted_label_eng, predicted_label_eng)
            print(f"Afrikaans Prediction: {af_label}")

            # Slide the window: remove oldest STEP_SIZE samples in-place
            for _ in range(STEP_SIZE):
                buffer.popleft()

        # Reduce CPU usage slightly
        time.sleep(0.01)

except KeyboardInterrupt:
    print("Real-time decoding stopped by user.")
