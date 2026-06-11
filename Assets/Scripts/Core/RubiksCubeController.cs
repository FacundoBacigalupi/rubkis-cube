using System.Collections.Generic;
using UnityEngine;

public class RubiksCubeController : MonoBehaviour
{
    public static RubiksCubeController Instance { get; private set; }

    [Header("References")]
    public LayerTurnAnimator layerAnimator;
    [SerializeField] public GameObject cubiePrefab;

    [Header("Settings")]
    public bool allowMiddleSlices = false;

    private Cubie[] cubies;
    private bool isAnimating = false;

    public bool IsAnimating => isAnimating;

    void Awake()
    {
        Instance = this;
        if (layerAnimator == null)
            layerAnimator = GetComponent<LayerTurnAnimator>();
    }

    void Start()
    {
        BuildCube();
    }

    void BuildCube()
    {
        cubies = new Cubie[27];
        int i = 0;
        for (int x = -1; x <= 1; x++)
        for (int y = -1; y <= 1; y++)
        for (int z = -1; z <= 1; z++)
        {
            var coord = new Vector3Int(x, y, z);
            var go    = Instantiate(cubiePrefab, transform);
            go.name   = $"Cubie({x},{y},{z})";

            var cubie = go.GetComponent<Cubie>();
            foreach (var sf in go.GetComponentsInChildren<StickerFace>(true))
            {
                sf.Cubie = cubie;
                if (!sf.TryGetComponent<RoundedStickerMesh>(out _))
                    sf.gameObject.AddComponent<RoundedStickerMesh>();
                sf.gameObject.SetActive(IsOutwardFace(coord, sf.LocalNormal));
            }

            cubie.SetCoord(coord);
            cubies[i++] = cubie;
        }
    }

    static bool IsOutwardFace(Vector3Int coord, Vector3Int normal)
    {
        if (normal.x != 0 && coord.x == normal.x) return true;
        if (normal.y != 0 && coord.y == normal.y) return true;
        if (normal.z != 0 && coord.z == normal.z) return true;
        return false;
    }

    // Destroys all cubies and rebuilds the cube in the fully-solved state.
    // Called before every shuffle so history always starts from solved.
    public void ResetToSolved()
    {
        if (cubies != null)
            foreach (var c in cubies)
                if (c != null) DestroyImmediate(c.gameObject);

        RubikMoveHistory.Instance.Clear();
        BuildCube();
    }

    public List<Cubie> GetCubiesOnLayer(RubikAxis axis, int layer)
    {
        var list = new List<Cubie>();
        foreach (var c in cubies)
        {
            if (GetCoordOnAxis(c.Coord, axis) == layer)
                list.Add(c);
        }
        return list;
    }

    public void ApplyMove(RubikMove move, bool addToHistory)
    {
        if (isAnimating) return;
        isAnimating = true;
        layerAnimator.AnimateMove(move, addToHistory, OnMoveComplete);
    }

    // Applies a move with no animation — used by the shuffler so the scramble is invisible.
    public void ApplyMoveInstant(RubikMove move, bool addToHistory)
    {
        Vector3 axisVec = move.Axis switch
        {
            RubikAxis.X => Vector3.right,
            RubikAxis.Y => Vector3.up,
            RubikAxis.Z => Vector3.forward,
            _           => Vector3.up
        };
        Quaternion rot = Quaternion.AngleAxis(90f * move.Direction, axisVec);

        foreach (var c in GetCubiesOnLayer(move.Axis, move.Layer))
        {
            c.transform.localPosition = rot * c.transform.localPosition;
            c.transform.localRotation = rot * c.transform.localRotation;
            SnapInstant(c.transform);
            c.UpdateCoord(RotateCoord(c.Coord, move.Axis, move.Direction));
        }

        if (addToHistory)
            RubikMoveHistory.Instance.Push(move);
    }

    private const float Spacing = 1.02f;

    private static void SnapInstant(Transform t)
    {
        Vector3 p = t.localPosition;
        t.localPosition = new Vector3(
            Mathf.Round(p.x / Spacing) * Spacing,
            Mathf.Round(p.y / Spacing) * Spacing,
            Mathf.Round(p.z / Spacing) * Spacing);
        Vector3 e = t.localEulerAngles;
        t.localEulerAngles = new Vector3(
            Mathf.Round(e.x / 90f) * 90f,
            Mathf.Round(e.y / 90f) * 90f,
            Mathf.Round(e.z / 90f) * 90f);
    }

    void OnMoveComplete(RubikMove move)
    {
        foreach (var c in GetCubiesOnLayer(move.Axis, move.Layer))
            c.UpdateCoord(RotateCoord(c.Coord, move.Axis, move.Direction));

        isAnimating = false;

        if (!ReverseHistorySolver.IsRunning && IsSolved())
            RubikUIController.Instance?.OnPlayerSolved();
    }

    public bool IsSolved()
    {
        foreach (var c in cubies)
        foreach (var sf in c.GetComponentsInChildren<StickerFace>())
            if (SnapToAxis(sf.transform.forward) != sf.LocalNormal)
                return false;
        return true;
    }

    static Vector3Int SnapToAxis(Vector3 v)
    {
        float ax = Mathf.Abs(v.x), ay = Mathf.Abs(v.y), az = Mathf.Abs(v.z);
        if (ax >= ay && ax >= az) return new Vector3Int((int)Mathf.Sign(v.x), 0, 0);
        if (ay >= ax && ay >= az) return new Vector3Int(0, (int)Mathf.Sign(v.y), 0);
        return new Vector3Int(0, 0, (int)Mathf.Sign(v.z));
    }

    public static int GetCoordOnAxis(Vector3Int coord, RubikAxis axis)
    {
        return axis switch
        {
            RubikAxis.X => coord.x,
            RubikAxis.Y => coord.y,
            RubikAxis.Z => coord.z,
            _ => 0
        };
    }

    public static Vector3Int RotateCoord(Vector3Int c, RubikAxis axis, int direction)
    {
        if (axis == RubikAxis.X && direction > 0) return new Vector3Int(c.x, -c.z,  c.y);
        if (axis == RubikAxis.X && direction < 0) return new Vector3Int(c.x,  c.z, -c.y);
        if (axis == RubikAxis.Y && direction > 0) return new Vector3Int( c.z, c.y, -c.x);
        if (axis == RubikAxis.Y && direction < 0) return new Vector3Int(-c.z, c.y,  c.x);
        if (axis == RubikAxis.Z && direction > 0) return new Vector3Int(-c.y, c.x,  c.z);
        if (axis == RubikAxis.Z && direction < 0) return new Vector3Int( c.y,-c.x,  c.z);
        return c;
    }

    public static RubikAxis AbsoluteAxisOf(Vector3Int v)
    {
        if (Mathf.Abs(v.x) > 0) return RubikAxis.X;
        if (Mathf.Abs(v.y) > 0) return RubikAxis.Y;
        return RubikAxis.Z;
    }

    public static int SignAlongAxis(Vector3Int v, RubikAxis axis)
    {
        return axis switch
        {
            RubikAxis.X => (int)Mathf.Sign(v.x),
            RubikAxis.Y => (int)Mathf.Sign(v.y),
            RubikAxis.Z => (int)Mathf.Sign(v.z),
            _ => 1
        };
    }

    public static Vector3Int Cross(Vector3Int a, Vector3Int b)
    {
        return new Vector3Int(
            a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x
        );
    }
}
