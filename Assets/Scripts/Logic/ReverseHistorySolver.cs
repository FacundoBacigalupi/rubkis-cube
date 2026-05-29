using System.Collections;
using UnityEngine;

public class ReverseHistorySolver : MonoBehaviour
{
    public static ReverseHistorySolver Instance { get; private set; }
    public static bool IsRunning { get; private set; }

    void Awake() => Instance = this;

    public void Solve()
    {
        StartCoroutine(SolveCoroutine());
    }

    private IEnumerator SolveCoroutine()
    {
        IsRunning = true;
        var history = RubikMoveHistory.Instance;
        var cube    = RubiksCubeController.Instance;
        var ui      = RubikUIController.Instance;

        ui?.SetStatus("Solving...");

        while (!history.IsEmpty)
        {
            cube.ApplyMove(history.Pop().Inverse(), addToHistory: false);
            yield return new WaitUntil(() => !cube.IsAnimating);
        }

        IsRunning = false;

        ui?.SetStatus("Solved!");
        yield return new WaitForSeconds(2f);
        ui?.SetStatus("");
    }
}
