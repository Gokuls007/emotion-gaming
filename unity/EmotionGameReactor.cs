using UnityEngine;

/// <summary>
/// Example of reacting to emotions in a game. Attach it next to an EmotionReceiver
/// (or assign one in the Inspector). It:
///  - shows the current emotion, mode, score and difficulty on screen;
///  - optionally tints a Light by game mode;
///  - exposes Difficulty so other scripts (enemy speed, spawn rate, ...) can scale with it.
/// </summary>
public class EmotionGameReactor : MonoBehaviour
{
    public EmotionReceiver receiver;

    [Tooltip("Optional: a light whose color follows the game mode")]
    public Light moodLight;

    [Tooltip("Show an on-screen status box")]
    public bool showOverlay = true;

    /// <summary>Difficulty multiplier from Python (1.0 until the first message).</summary>
    public float Difficulty { get; private set; }

    /// <summary>Current game mode from Python ("normal" until the first message).</summary>
    public string GameState { get; private set; }

    private EmotionData last;

    private void Awake()
    {
        Difficulty = 1f;
        GameState = "normal";
        if (receiver == null) receiver = GetComponent<EmotionReceiver>();
    }

    private void OnEnable()
    {
        if (receiver != null) receiver.EmotionReceived += OnEmotion;
        else Debug.LogWarning("EmotionGameReactor: no EmotionReceiver assigned.");
    }

    private void OnDisable()
    {
        if (receiver != null) receiver.EmotionReceived -= OnEmotion;
    }

    private void OnEmotion(EmotionData data)
    {
        last = data;
        Difficulty = data.difficulty;
        GameState = data.game_state;
        if (moodLight != null) moodLight.color = ColorFor(data.game_state);
    }

    public static Color ColorFor(string gameState)
    {
        switch (gameState)
        {
            case "bonus_mode": return Color.yellow;                  // happy
            case "battle_mode": return Color.red;                    // angry
            case "mystery_mode": return Color.magenta;               // surprise
            case "stealth_mode": return new Color(0.2f, 0.2f, 0.5f); // fear
            case "comfort_mode": return Color.cyan;                  // sad
            case "defense_mode": return Color.green;                 // disgust
            default: return Color.white;                             // normal / neutral
        }
    }

    private void OnGUI()
    {
        if (!showOverlay) return;
        string text = last == null
            ? "Waiting for emotion_game_unity.py on UDP port " + (receiver != null ? receiver.port : 5065) + "..."
            : "Emotion: " + last.emotion + " (" + last.confidence.ToString("0.0") + "%)\n" +
              "Mode: " + last.game_state + "\n" +
              "Score: " + last.score + "\n" +
              "Difficulty: " + last.difficulty.ToString("0.00");
        GUI.Box(new Rect(10, 10, 320, 90), text);
    }
}
