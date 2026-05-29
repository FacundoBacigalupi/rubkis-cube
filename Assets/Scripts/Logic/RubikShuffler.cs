using UnityEngine;

public class RubikShuffler : MonoBehaviour
{
    public static RubikShuffler Instance { get; private set; }

    [SerializeField] private int shuffleCount = 25;

    private static readonly RubikAxis[] Axes       = { RubikAxis.X, RubikAxis.Y, RubikAxis.Z };
    private static readonly int[]       OuterLayers = { -1, 1 };
    private static readonly int[]       Directions  = { -1, 1 };

    void Awake() => Instance = this;

    public void Shuffle()
    {
        var cube = RubiksCubeController.Instance;
        cube.ResetToSolved();   // always start from solved — guarantees Solve returns to solved

        RubikMove? lastMove = null;

        for (int i = 0; i < shuffleCount; i++)
        {
            RubikMove move;
            int tries = 0;
            do
            {
                var axis  = Axes[Random.Range(0, Axes.Length)];
                int layer = OuterLayers[Random.Range(0, OuterLayers.Length)];
                int dir   = Directions[Random.Range(0, Directions.Length)];
                move = new RubikMove(axis, layer, dir);
                tries++;
            }
            // Reject move that would immediately undo the previous one (same axis+layer, opposite dir)
            while (tries < 20 && lastMove.HasValue && Cancels(move, lastMove.Value));

            cube.ApplyMoveInstant(move, addToHistory: true);
            lastMove = move;
        }

        RubikUIController.Instance?.StartTimer();
    }

    private static bool Cancels(RubikMove a, RubikMove b)
        => a.Axis == b.Axis && a.Layer == b.Layer && a.Direction != b.Direction;
}
