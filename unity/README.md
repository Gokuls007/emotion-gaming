# Unity integration

`emotion_game_unity.py` sends one JSON message over UDP to `127.0.0.1:5065` whenever the
detected emotion changes the game state or score:

```json
{"emotion": "happy", "confidence": 97.34, "score": 100, "difficulty": 1.3,
 "game_state": "bonus_mode", "timestamp": 1790556808.0}
```

The two scripts in this folder receive those messages in Unity.

| Script | What it does |
|---|---|
| `EmotionReceiver.cs` | Listens on UDP port 5065 on a background thread, parses each message with `JsonUtility`, and raises the `EmotionReceived` event on Unity's main thread. The newest message is always in `Latest`. |
| `EmotionGameReactor.cs` | Example game logic: an on-screen status box, an optional `Light` tinted by game mode, and a `Difficulty` value other scripts can read. |

## Setup

1. Copy both `.cs` files into your Unity project's `Assets/` folder (for example `Assets/Scripts/`).
2. In your scene, create an empty GameObject named `EmotionManager`.
3. Add **EmotionReceiver** and **EmotionGameReactor** to it. You can drag a scene Light into
   *Mood Light* if you want the colour effect.
4. Press **Play**. The Console shows `EmotionReceiver: listening on UDP port 5065`.
5. In a terminal, from the project root:
   ```bash
   python emotion_game_unity.py
   ```
   Choose **1** (Unity mode) and press Enter. Look at the webcam: the on-screen box and the
   Console update as emotions are detected.

## Using the data in your game

```csharp
public class EnemySpeed : MonoBehaviour
{
    public EmotionReceiver receiver;
    public float baseSpeed = 3f;

    void OnEnable()  { receiver.EmotionReceived += OnEmotion; }
    void OnDisable() { receiver.EmotionReceived -= OnEmotion; }

    void OnEmotion(EmotionData data)
    {
        // data.emotion, data.confidence, data.score, data.difficulty, data.game_state
        GetComponent<UnityEngine.AI.NavMeshAgent>().speed = baseSpeed * data.difficulty;
    }
}
```

| Emotion | game_state | Difficulty |
|---|---|---|
| happy | bonus_mode | 1.3 |
| angry | battle_mode | 1.5 |
| surprise | mystery_mode | unchanged |
| fear | stealth_mode | 0.8 |
| sad | comfort_mode | 0.7 |
| disgust | defense_mode | unchanged |
| neutral | normal | unchanged |

## Troubleshooting

- **"could not listen on UDP port 5065"**: another program (or a second copy of the
  receiver) is already using the port. Close it, or change `port` on the component and
  `unity_port` in `emotion_game_unity.py` to the same new number.
- **Nothing arrives**: start Play in Unity *before* choosing option 1 in Python. Python only
  sends when the game state or score changes. Emotions in the same mode still add points,
  but `neutral` after `normal` sends nothing.
- **Both on one PC** needs no firewall change (`127.0.0.1`). To run Unity on another machine,
  pass its IP as `unity_ip` in `emotion_game_unity.py` and allow UDP 5065 in that machine's
  firewall.
