"""
WebSocket middleware between the Unity client and the SER model.

Protocol
--------
Client -> server
  * (optional, text) JSON config, sent once after connecting:
        {"sample_rate": 48000}
    If omitted, audio is assumed to be 16000 Hz.
  * (binary) raw audio: little-endian signed 16-bit PCM, mono, at the
    declared sample rate. Any chunk size works (Unity sends ~100 ms).

Server -> all connected clients (text JSON), roughly once per --hop seconds:
    {"emotion": "Stressed",        # current state (smoothed + threshold)
     "raw_emotion": "Stressed",    # this window's argmax alone
     "confidence": 0.82,           # this window's top probability
     "probabilities": {"Neutral": 0.05, "Stressed": 0.82, "Confident": 0.13},
     "smoothed": {...},            # average of the last --smooth windows
     "smoothed_confidence": 0.74,  # what the threshold is applied to
     "state_changed": true,
     "silent": false,
     "warming_up": false,          # true during the first --warmup seconds
     "rms": 0.045,
     "timestamp": 1727512345.12}

Every connected client receives every result, so the Unity app, the
instructor dashboard, and a logger can all listen on the same socket.

Run (real model):
    python server.py --model ../models/ser_cnn_lstm.h5 \
        --norm ../features/norm_stats.npy --labels ../models/label_encoder.pkl

Run (no model - cycles fake emotions, for testing the Unity side):
    python server.py --dummy
"""
import argparse
import asyncio
import json
import math
import threading
import time

import numpy as np
import websockets
from scipy.signal import resample_poly

from inference import EmotionPredictor, load_artifacts
from label_map import TARGET_CLASSES
from preprocess import SAMPLE_RATE

clients = set()
predict_lock = threading.Lock()   # one model.predict at a time


class DummyModel:
    """Stand-in model: returns a confident prediction that cycles through the
    classes every 3 calls. Lets you test Unity <-> server plumbing."""

    def __init__(self, n_classes):
        self.n, self.calls = n_classes, 0

    def predict(self, x, verbose=0):
        target = (self.calls // 3) % self.n
        self.calls += 1
        probs = np.full(self.n, 0.05)
        probs[target] = 1.0 - 0.05 * (self.n - 1)
        return probs[np.newaxis]


def to_16k(audio, sr):
    if sr == SAMPLE_RATE:
        return audio
    g = math.gcd(SAMPLE_RATE, sr)
    return resample_poly(audio, SAMPLE_RATE // g, sr // g).astype(np.float32)


def make_handler(args, model, mean, std, classes):
    hop_samples = int(SAMPLE_RATE * args.hop)

    async def handler(ws):
        loop = asyncio.get_running_loop()
        predictor = EmotionPredictor(model, mean, std, classes,
                                     threshold=args.threshold,
                                     silence_rms=args.silence_rms,
                                     smooth_windows=args.smooth,
                                     warmup_sec=args.warmup)
        sample_rate, pending = SAMPLE_RATE, 0
        clients.add(ws)
        print(f"[+] client connected ({len(clients)} total)")
        try:
            async for msg in ws:
                if isinstance(msg, str):
                    try:
                        sample_rate = int(json.loads(msg).get("sample_rate", sample_rate))
                        print(f"    client sample_rate = {sample_rate}")
                    except (ValueError, AttributeError):
                        pass
                    continue

                pcm = np.frombuffer(msg, dtype="<i2").astype(np.float32) / 32768.0
                audio = to_16k(pcm, sample_rate)
                predictor.push_audio(audio)
                pending += len(audio)

                if pending >= hop_samples:
                    pending = 0

                    def run():
                        with predict_lock:
                            return predictor.predict()

                    result = await loop.run_in_executor(None, run)
                    result["timestamp"] = time.time()
                    websockets.broadcast(clients, json.dumps(result))
                    if result["warming_up"]:
                        tag = "warming up"
                    elif result["silent"]:
                        tag = f'silent, rms={result["rms"]:.4f}'
                    else:
                        tag = (f'raw {result["raw_emotion"]} {result["confidence"]:.2f}, '
                               f'smoothed {result["smoothed_confidence"]:.2f}, '
                               f'rms={result["rms"]:.4f}')
                    print(f'    state={result["emotion"]:<10} ({tag})'
                          f'{"  <- changed" if result["state_changed"] else ""}')
        except websockets.ConnectionClosed:
            pass
        finally:
            clients.discard(ws)
            print(f"[-] client disconnected ({len(clients)} total)")

    return handler


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="../models/ser_cnn_lstm.h5")
    ap.add_argument("--norm", default="../features/norm_stats.npy")
    ap.add_argument("--labels", default="../models/label_encoder.pkl")
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 lets other devices (e.g. a Quest) connect")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--threshold", type=float, default=0.6,
                    help="min top-class probability to change emotion state")
    ap.add_argument("--hop", type=float, default=1.0,
                    help="seconds of new audio between predictions")
    ap.add_argument("--silence_rms", type=float, default=0.02,
                    help="below this RMS the window counts as silence. Measure "
                         "your mic with mic_check.py; recorded RAVDESS clips "
                         "need about 0.002")
    ap.add_argument("--smooth", type=int, default=3,
                    help="number of recent windows averaged before deciding "
                         "(1 = no smoothing)")
    ap.add_argument("--warmup", type=float, default=1.0,
                    help="seconds of audio ignored after a client connects "
                         "(skips the mic start-up glitch)")
    ap.add_argument("--dummy", action="store_true",
                    help="use a fake model (no TensorFlow needed)")
    args = ap.parse_args()

    if args.dummy:
        classes = TARGET_CLASSES
        model, mean, std = DummyModel(len(classes)), 0.0, 1.0
        print("Running with DUMMY model - predictions are fake.")
    else:
        model, mean, std, classes = load_artifacts(args.model, args.norm, args.labels)
        print(f"Loaded model. Classes: {classes}")

    handler = make_handler(args, model, mean, std, classes)
    async with websockets.serve(handler, args.host, args.port,
                                max_size=2 ** 20):
        print(f"SER server listening on ws://{args.host}:{args.port} "
              f"(threshold={args.threshold}, hop={args.hop}s, "
              f"silence_rms={args.silence_rms}, smooth={args.smooth}, "
              f"warmup={args.warmup}s)")
        await asyncio.Future()


if __name__ == "__main__":
    asyncio.run(main())
