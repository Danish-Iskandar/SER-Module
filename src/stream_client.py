"""
Test client for server.py - lets you verify the whole real-time path
without Unity.

Stream a recorded clip in real time (e.g. a RAVDESS file):
    python stream_client.py --wav path/to/03-01-06-01-01-01-05.wav

Stream your laptop microphone (needs: pip install sounddevice):
    python stream_client.py --mic
"""
import argparse
import asyncio
import json

import numpy as np
import websockets

CHUNK_SEC = 0.1


async def receive(ws):
    async for msg in ws:
        r = json.loads(msg)
        if r.get("warming_up"):
            print(f'  state={r["emotion"]:<10} (warming up)')
        elif r["silent"]:
            print(f'  state={r["emotion"]:<10} (silence, rms={r["rms"]:.4f})')
        else:
            sm = "  ".join(f"{k[:4]}={v:.2f}" for k, v in r["smoothed"].items())
            print(f'  state={r["emotion"]:<10} raw={r["raw_emotion"]:<10} '
                  f'conf={r["confidence"]:.2f}  smoothed[{sm}]  rms={r["rms"]:.3f}'
                  f'{"  <- changed" if r["state_changed"] else ""}')


async def send_wav(ws, path):
    import soundfile as sf
    data, sr = sf.read(path, dtype="int16")
    if data.ndim > 1:
        data = data[:, 0]
    await ws.send(json.dumps({"sample_rate": int(sr)}))
    step = int(sr * CHUNK_SEC)
    print(f"Streaming {path} ({len(data) / sr:.1f}s @ {sr} Hz) in real time...")
    for i in range(0, len(data), step):
        await ws.send(data[i:i + step].astype("<i2").tobytes())
        await asyncio.sleep(CHUNK_SEC)
    # trailing silence so the last window gets flushed through
    await asyncio.sleep(1.5)


async def send_mic(ws, sr=16000):
    import sounddevice as sd
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()

    def callback(indata, frames, t, status):
        loop.call_soon_threadsafe(queue.put_nowait, indata.copy())

    await ws.send(json.dumps({"sample_rate": sr}))
    print("Listening on the default microphone - speak! (Ctrl+C to stop)")
    with sd.InputStream(samplerate=sr, channels=1, dtype="int16",
                        blocksize=int(sr * CHUNK_SEC), callback=callback):
        while True:
            chunk = await queue.get()
            await ws.send(chunk.astype("<i2").tobytes())


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://localhost:8765")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--wav")
    g.add_argument("--mic", action="store_true")
    args = ap.parse_args()

    async with websockets.connect(args.url) as ws:
        recv = asyncio.create_task(receive(ws))
        try:
            await (send_mic(ws) if args.mic else send_wav(ws, args.wav))
        finally:
            recv.cancel()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
