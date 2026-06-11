using UnityEngine;
using UnityEngine.UI;

public class RubikUIController : MonoBehaviour
{
    public static RubikUIController Instance { get; private set; }

    [SerializeField] private Button shuffleButton;
    [SerializeField] private Button solveButton;
    [SerializeField] private Button neuralSolveButton;
    [SerializeField] private Text   statusText;

    private float timerSeconds;
    private bool  timerRunning;

    void Awake() => Instance = this;

    void Start()
    {
        shuffleButton.onClick.AddListener(OnShuffle);
        solveButton.onClick.AddListener(OnSolve);
        if (neuralSolveButton != null)
            neuralSolveButton.onClick.AddListener(OnNeuralSolve);
    }

    void Update()
    {
        if (!timerRunning) return;
        timerSeconds += Time.deltaTime;
        SetStatus(FormatTime(timerSeconds));
    }

    void OnShuffle()
    {
        if (RubiksCubeController.Instance.IsAnimating) return;
        RubikShuffler.Instance.Shuffle();
    }

    void OnSolve()
    {
        if (RubiksCubeController.Instance.IsAnimating) return;
        if (RubikMoveHistory.Instance.IsEmpty) return;
        StopTimer();
        SetStatus("");
        ReverseHistorySolver.Instance.Solve();
    }

    void OnNeuralSolve()
    {
        if (RubiksCubeController.Instance.IsAnimating) return;
        if (NeuralSolver.IsRunning) return;
        StopTimer();
        SetStatus("");
        NeuralSolver.Instance.Solve();
    }

    // Called by RubiksCubeController when the player manually solves the cube.
    public void OnPlayerSolved()
    {
        if (!timerRunning) return;
        float time = timerSeconds;
        StopTimer();
        SetStatus($"Solved! {FormatTime(time)}");
    }

    public void StartTimer()
    {
        timerSeconds = 0f;
        timerRunning = true;
        SetStatus(FormatTime(0f));
    }

    public void StopTimer()
    {
        timerRunning = false;
    }

    public float TimerSeconds => timerSeconds;

    public void SetStatus(string msg)
    {
        if (statusText != null)
            statusText.text = msg;
    }

    public static string FormatTime(float seconds)
    {
        int min = (int)(seconds / 60);
        int sec = (int)(seconds % 60);
        int dec = (int)((seconds % 1f) * 10);
        return $"{min:D2}:{sec:D2}.{dec}";
    }
}
