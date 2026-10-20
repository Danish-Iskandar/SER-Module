// SERClient.cs
// Streams microphone audio to the Python SER server over WebSocket and
// receives emotion state updates back (see src/server.py for the protocol).
//
// SETUP
//  1. Install NativeWebSocket: Window > Package Manager > "+" > Add package
//     from git URL:  https://github.com/endel/NativeWebSocket.git#upm
//  2. Add this script to any GameObject (e.g. an empty "SER" object).
//  3. Drag the avatar's Renderer into "Avatar Renderer".
//  4. Set "Server Url":
//       - Unity Editor on the same PC as the server:  ws://localhost:8765
//       - Standalone Quest: ws://<PC's LAN IP>:8765 (same Wi-Fi; allow the
//         port through the PC firewall)
//  5. Quest builds only: Player Settings > Android will add the microphone
//     permission automatically once Microphone is used; this script also
//     requests it at runtime.
//
// The server declares/handles resampling, so we simply send whatever sample
// rate the device microphone gives us.

using System;
using System.Collections;
using UnityEngine;
using UnityEngine.Events;
using NativeWebSocket;
#if UNITY_ANDROID
using UnityEngine.Android;
#endif

[Serializable]
public class EmotionResult
{
    public string emotion;        // current (threshold-filtered) state
    public string raw_emotion;    // this window's argmax (null when silent)
    public float confidence;            // this window alone
    public float smoothed_confidence;   // averaged over recent windows (drives the state)
    public bool state_changed;
    public bool silent;
    public bool warming_up;
    public double timestamp;
}

[Serializable]
public class EmotionEvent : UnityEvent<string, float> { }

public class SERClient : MonoBehaviour
{
    [Header("Connection")]
    public string serverUrl = "ws://localhost:8765";

    [Header("Microphone")]
    [Tooltip("Requested rate; the device may give a different one (handled).")]
    public int requestedSampleRate = 16000;
    [Tooltip("Seconds of audio per WebSocket message.")]
    public float chunkSeconds = 0.1f;

    [Header("Avatar feedback")]
    public Renderer avatarRenderer;
    public Color neutralColor = new Color(0.75f, 0.75f, 0.75f);
    public Color stressedColor = Color.red;
    public Color confidentColor = new Color(0.2f, 0.8f, 0.3f);
    public float colorLerpSpeed = 4f;

    [Header("Hook the adaptive engine here")]
    [Tooltip("Fires when the emotion STATE changes: (emotion, confidence).")]
    public EmotionEvent onEmotionChanged;

    public string CurrentEmotion { get; private set; } = "Neutral";

    WebSocket ws;
    AudioClip micClip;
    int micSampleRate;
    int lastMicPos;
    bool micReady;   // true only after the sample-rate config has been sent
    Color targetColor;

    async void Start()
    {
        targetColor = neutralColor;

        ws = new WebSocket(serverUrl);
        ws.OnOpen += () =>
        {
            Debug.Log("[SER] connected to " + serverUrl);
            StartCoroutine(StartMicrophone());
        };
        ws.OnError += (e) => Debug.LogWarning("[SER] error: " + e);
        ws.OnClose += (c) => Debug.Log("[SER] closed: " + c);
        ws.OnMessage += (bytes) =>
        {
            var json = System.Text.Encoding.UTF8.GetString(bytes);
            HandleResult(JsonUtility.FromJson<EmotionResult>(json));
        };

        await ws.Connect();   // returns when the socket closes
    }

    IEnumerator StartMicrophone()
    {
#if UNITY_ANDROID
        if (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
        {
            Permission.RequestUserPermission(Permission.Microphone);
            while (!Permission.HasUserAuthorizedPermission(Permission.Microphone))
                yield return null;
        }
#endif
        if (Microphone.devices.Length == 0)
        {
            Debug.LogError("[SER] no microphone found");
            yield break;
        }

        // Looping 10 s ring buffer on the default device.
        micClip = Microphone.Start(null, true, 10, requestedSampleRate);
        while (Microphone.GetPosition(null) <= 0) yield return null;

        micSampleRate = micClip.frequency;
        lastMicPos = 0;
        ws.SendText("{\"sample_rate\": " + micSampleRate + "}");
        micReady = true;   // audio may only flow AFTER the config message above
        Debug.Log("[SER] microphone started at " + micSampleRate + " Hz");
    }

    void Update()
    {
#if !UNITY_WEBGL || UNITY_EDITOR
        ws?.DispatchMessageQueue();
#endif
        StreamMicrophone();

        if (avatarRenderer != null)
        {
            avatarRenderer.material.color = Color.Lerp(
                avatarRenderer.material.color, targetColor,
                Time.deltaTime * colorLerpSpeed);
        }
    }

    void StreamMicrophone()
    {
        if (!micReady || micClip == null || ws == null || ws.State != WebSocketState.Open) return;

        int pos = Microphone.GetPosition(null);
        if (pos < 0) return;

        int available = pos - lastMicPos;
        if (available < 0) available += micClip.samples;   // ring buffer wrapped

        int chunk = Mathf.Max(1, (int)(micSampleRate * chunkSeconds));
        while (available >= chunk)
        {
            float[] samples = ReadRing(lastMicPos, chunk);
            lastMicPos = (lastMicPos + chunk) % micClip.samples;
            available -= chunk;
            _ = ws.Send(ToPcm16(samples));
        }
    }

    // Reads `count` samples starting at `start`, handling ring-buffer wrap.
    float[] ReadRing(int start, int count)
    {
        var data = new float[count];
        int total = micClip.samples;
        if (start + count <= total)
        {
            micClip.GetData(data, start);
        }
        else
        {
            int first = total - start;
            var a = new float[first];
            var b = new float[count - first];
            micClip.GetData(a, start);
            micClip.GetData(b, 0);
            Array.Copy(a, 0, data, 0, first);
            Array.Copy(b, 0, data, first, b.Length);
        }
        return data;
    }

    // float [-1,1] -> little-endian signed 16-bit PCM bytes
    static byte[] ToPcm16(float[] samples)
    {
        var pcm = new short[samples.Length];
        for (int i = 0; i < samples.Length; i++)
            pcm[i] = (short)Mathf.Clamp(samples[i] * 32767f, -32768f, 32767f);
        var bytes = new byte[pcm.Length * 2];
        Buffer.BlockCopy(pcm, 0, bytes, 0, bytes.Length);   // LE on Windows/Android
        return bytes;
    }

    void HandleResult(EmotionResult r)
    {
        if (r == null || string.IsNullOrEmpty(r.emotion)) return;

        CurrentEmotion = r.emotion;
        switch (r.emotion)
        {
            case "Stressed":  targetColor = stressedColor;  break;
            case "Confident": targetColor = confidentColor; break;
            default:          targetColor = neutralColor;   break;
        }

        if (r.state_changed)
        {
            Debug.Log($"[SER] state -> {r.emotion} ({r.smoothed_confidence:F2})");
            onEmotionChanged?.Invoke(r.emotion, r.smoothed_confidence);
        }
    }

    async void OnDestroy()
    {
        if (Microphone.IsRecording(null)) Microphone.End(null);
        if (ws != null) await ws.Close();
    }
}
