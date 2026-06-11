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
        cube.ResetToSolved();

        RubikAxis? lastAxis = null;

        // Build axis list once; each step we pick only from the two axes != lastAxis
        var allAxes = new[] { RubikAxis.X, RubikAxis.Y, RubikAxis.Z };

        for (int i = 0; i < shuffleCount; i++)
        {
            // Only allow axes perpendicular to the previous move
            RubikAxis[] candidates = lastAxis.HasValue
                ? System.Array.FindAll(allAxes, a => a != lastAxis.Value)
                : allAxes;

            var axis  = candidates[Random.Range(0, candidates.Length)];
            int layer = OuterLayers[Random.Range(0, OuterLayers.Length)];
            int dir   = Directions[Random.Range(0, Directions.Length)];
            var move  = new RubikMove(axis, layer, dir);

            cube.ApplyMoveInstant(move, addToHistory: true);
            lastAxis = axis;
        }

        RubikUIController.Instance?.StartTimer();
    }
}
