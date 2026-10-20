"""
Measures your microphone's noise and speech levels and suggests a
--silence_rms value for server.py.

    python mic_check.py              # default input device
    python mic_check.py --device 3   # a specific device from the printed list

Records the same way stream_client.py does (16 kHz, 16-bit). A 3-second
throwaway recording runs first, because many microphones emit a burst of
garbage when the stream opens.
"""
import argparse

import numpy as np
import sounddevice as sd

SR = 16000
SECONDS = 3


def record(device):
    x = sd.rec(SR * SECONDS, samplerate=SR, channels=1, dtype="int16",
               device=device)
    sd.wait()
    return x[:, 0].astype(np.float32) / 32768


def describe(x):
    return (f"rms={np.std(x):.4f}  peak={np.abs(x).max():.4f}  "
            f"dc_offset={x.mean():.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None)
    args = ap.parse_args()

    print(sd.query_devices())
    dev = args.device if args.device is not None else sd.default.device[0]
    print(f"\nUsing input device {dev}\n")

    print("Warming up the microphone (3 s)...")
    warm = record(args.device)
    print(f"  start-up recording: {describe(warm)}")

    input("\nStay QUIET: press Enter, then stay silent for 3 s...")
    quiet = record(args.device)
    print(f"  quiet:   {describe(quiet)}")

    input("\nTALK normally: press Enter, then keep talking for 3 s...")
    talk = record(args.device)
    print(f"  talking: {describe(talk)}")

    q, t = float(np.std(quiet)), float(np.std(talk))
    print()
    if t < q * 2:
        print("Talking is not clearly louder than quiet. Python is probably "
              "using the wrong input device, or the input volume is very low. "
              "Check GNOME Settings > Sound > Input, or rerun with --device N.")
    else:
        # Geometric mean: halfway between the two on a log scale.
        suggestion = float(np.sqrt(q * t))
        print(f"Suggested: python server.py ... --silence_rms {suggestion:.3f}")


if __name__ == "__main__":
    main()
