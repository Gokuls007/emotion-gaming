using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Threading;
using UnityEngine;

/// <summary>
/// One message from emotion_game_unity.py (JSON sent over UDP).
/// Field names must match the Python keys exactly for JsonUtility.
/// </summary>
[Serializable]
public class EmotionData
{
    public string emotion;      // happy, sad, angry, surprise, fear, disgust, neutral
    public float confidence;    // 0-100
    public int score;
    public float difficulty;    // e.g. 1.3 in bonus mode, 0.7 in comfort mode
    public string game_state;   // bonus_mode, comfort_mode, battle_mode, mystery_mode,
                                // stealth_mode, defense_mode, normal
    public double timestamp;    // Unix time (seconds) when Python sent it
}

/// <summary>
/// Receives emotion updates from emotion_game_unity.py on a UDP port (default 5065).
///
/// Add this component to any GameObject in your scene, press Play, then run
/// "python emotion_game_unity.py" and choose option 1. Other scripts can read
/// <see cref="Latest"/> or subscribe to <see cref="EmotionReceived"/>.
///
/// Network I/O runs on a background thread; the event is raised on Unity's main
/// thread (from Update), so handlers may safely touch GameObjects.
/// </summary>
public class EmotionReceiver : MonoBehaviour
{
    [Tooltip("Must match unity_port in emotion_game_unity.py")]
    public int port = 5065;

    [Tooltip("Log every message to the Console")]
    public bool logMessages = true;

    /// <summary>Raised on the main thread for every message received.</summary>
    public event Action<EmotionData> EmotionReceived;

    /// <summary>The most recent message, or null before the first one arrives.</summary>
    public EmotionData Latest { get; private set; }

    /// <summary>Total messages received since Play started.</summary>
    public int MessageCount { get; private set; }

    private UdpClient client;
    private Thread receiveThread;
    private volatile bool running;
    private readonly object queueLock = new object();
    private readonly System.Collections.Generic.Queue<string> pending =
        new System.Collections.Generic.Queue<string>();

    private void OnEnable()
    {
        try
        {
            client = new UdpClient(port);
        }
        catch (SocketException e)
        {
            Debug.LogError("EmotionReceiver: could not listen on UDP port " + port +
                           " (is another program using it?): " + e.Message);
            enabled = false;
            return;
        }

        running = true;
        receiveThread = new Thread(ReceiveLoop);
        receiveThread.IsBackground = true;
        receiveThread.Start();
        Debug.Log("EmotionReceiver: listening on UDP port " + port);
    }

    private void ReceiveLoop()
    {
        IPEndPoint remote = new IPEndPoint(IPAddress.Any, 0);
        while (running)
        {
            try
            {
                byte[] data = client.Receive(ref remote);
                string json = Encoding.UTF8.GetString(data);
                lock (queueLock)
                {
                    pending.Enqueue(json);
                }
            }
            catch (SocketException)
            {
                // Thrown when the socket is closed in OnDisable; exit quietly.
                if (!running) return;
            }
            catch (ObjectDisposedException)
            {
                return;
            }
        }
    }

    private void Update()
    {
        while (true)
        {
            string json;
            lock (queueLock)
            {
                if (pending.Count == 0) break;
                json = pending.Dequeue();
            }
            Handle(json);
        }
    }

    private void Handle(string json)
    {
        EmotionData data;
        try
        {
            data = JsonUtility.FromJson<EmotionData>(json);
        }
        catch (Exception e)
        {
            Debug.LogWarning("EmotionReceiver: ignoring malformed message: " + json + " (" + e.Message + ")");
            return;
        }
        if (data == null || string.IsNullOrEmpty(data.emotion))
        {
            Debug.LogWarning("EmotionReceiver: ignoring message without an emotion: " + json);
            return;
        }

        Latest = data;
        MessageCount++;
        if (logMessages)
        {
            Debug.Log("EmotionReceiver: " + data.emotion + " (" + data.confidence.ToString("0.0") +
                      "%) state=" + data.game_state + " score=" + data.score +
                      " difficulty=" + data.difficulty.ToString("0.00"));
        }

        Action<EmotionData> handler = EmotionReceived;
        if (handler != null) handler(data);
    }

    private void OnDisable()
    {
        running = false;
        if (client != null)
        {
            client.Close();
            client = null;
        }
        if (receiveThread != null)
        {
            receiveThread.Join(500);
            receiveThread = null;
        }
    }
}
