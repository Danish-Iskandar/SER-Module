// SERListenerTest.cs
// Minimal test listener for SERClient's onEmotionChanged event.
// Use it to confirm emotion changes reach your own code before building the
// adaptive engine. Remove it (or replace it) once the real engine exists.
//
// SETUP
//  1. Add this component to any GameObject (e.g. the same "SER" object).
//  2. On the SERClient component, under "On Emotion Changed", click "+".
//  3. Drag the GameObject holding this script into the empty slot.
//  4. In the function dropdown pick:
//       SERListenerTest > Dynamic string, float > OnEmotionChanged
//     (choose the entry under "Dynamic", not "Static", so the real values
//     from SERClient are passed through).

using UnityEngine;

public class SERListenerTest : MonoBehaviour
{
    [Tooltip("Show the current emotion in the top-left corner of the Game view.")]
    public bool showOnScreen = true;

    string lastEmotion = "(none yet)";
    float lastConfidence;
    int changeCount;

    public void OnEmotionChanged(string emotion, float confidence)
    {
        lastEmotion = emotion;
        lastConfidence = confidence;
        changeCount++;
        Debug.Log($"[SER listener] #{changeCount}: {emotion} ({confidence:F2}) at {Time.time:F1}s");

        // This is where the adaptive engine will react, e.g.:
        // if (emotion == "Stressed") { /* ease the scenario */ }
    }

    void OnGUI()
    {
        if (!showOnScreen) return;
        GUI.Label(new Rect(10, 10, 400, 25),
            $"SER: {lastEmotion} ({lastConfidence:F2})  changes: {changeCount}");
    }
}
