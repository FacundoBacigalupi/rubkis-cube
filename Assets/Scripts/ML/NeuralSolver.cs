using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;


/// <summary>
/// Neural network solver using Unity InferenceEngine.
///
/// Greedy mode  — single forward pass per step, picks highest-logit move.
///
/// Beam mode    — maintains beamSize parallel candidate paths.
///   • If a policy+value ONNX is assigned to pvModelAsset, uses distance
///     estimates (value head) for scoring, which greatly improves success on
///     deep scrambles.
///   • Otherwise falls back to policy-probability-only scoring.
///
/// Scoring formula (value-guided):
///   score = pathLengthWeight * pathLen + valueWeight * nextValue
///           − policyWeight  * log(policy[move])
///   Lower is better.  nextValue = 0 at solved, 1 at maximum depth.
/// </summary>
public class NeuralSolver : MonoBehaviour
{
    public enum SolveMode { Greedy, Beam }

    public static NeuralSolver Instance { get; private set; }
    public static bool IsRunning { get; private set; }

    [Header("Models")]
    [Tooltip("Policy-only ONNX — required for Greedy mode and as fallback for Beam.")]
    [SerializeField] Unity.InferenceEngine.ModelAsset modelAsset;

    [Tooltip("Policy+value ONNX exported with --include-value. Optional but strongly " +
             "recommended for Beam mode on scrambles deeper than ~6 moves.")]
    [SerializeField] Unity.InferenceEngine.ModelAsset pvModelAsset;

    [Header("Settings")]
    [SerializeField] SolveMode solveMode    = SolveMode.Beam;
    [SerializeField] int maxGreedySteps     = 50;
    [SerializeField] int beamSize           = 64;
    [SerializeField] int beamTopK           = 5;
    [SerializeField] int maxBeamDepth       = 50;

    [Header("Beam scoring weights (value-guided only)")]
    [SerializeField] float pathLengthWeight = 0.1f;
    [SerializeField] float valueWeight      = 1.0f;
    [SerializeField] float policyWeight     = 0.1f;

    // ── move table ──────────────────────────────────────────────────────────
    // Order matches Python MOVES: R R' L L' U U' D D' F F' B B'

    static readonly RubikMove[] MoveTable =
    {
        new RubikMove(RubikAxis.X,  1, -1),  // 0  R
        new RubikMove(RubikAxis.X,  1,  1),  // 1  R'
        new RubikMove(RubikAxis.X, -1,  1),  // 2  L
        new RubikMove(RubikAxis.X, -1, -1),  // 3  L'
        new RubikMove(RubikAxis.Y,  1, -1),  // 4  U
        new RubikMove(RubikAxis.Y,  1,  1),  // 5  U'
        new RubikMove(RubikAxis.Y, -1,  1),  // 6  D
        new RubikMove(RubikAxis.Y, -1, -1),  // 7  D'
        new RubikMove(RubikAxis.Z,  1, -1),  // 8  F
        new RubikMove(RubikAxis.Z,  1,  1),  // 9  F'
        new RubikMove(RubikAxis.Z, -1,  1),  // 10 B
        new RubikMove(RubikAxis.Z, -1, -1),  // 11 B'
    };

    static readonly int[] InverseIndex = { 1,0, 3,2, 5,4, 7,6, 9,8, 11,10 };

    // ── workers and scratch buffers ─────────────────────────────────────────

    Unity.InferenceEngine.Worker _worker;    // policy-only
    Unity.InferenceEngine.Worker _pvWorker;  // policy+value (optional)

    readonly float[] _inputBuffer  = new float[324];
    readonly float[] _logitsBuffer = new float[12];

    // ── lifecycle ───────────────────────────────────────────────────────────

    void Awake() => Instance = this;

    void OnDestroy()
    {
        _worker?.Dispose();
        _pvWorker?.Dispose();
    }

    void EnsureWorkers()
    {
        if (_worker == null && modelAsset != null)
        {
            var model = Unity.InferenceEngine.ModelLoader.Load(modelAsset);
            _worker = new Unity.InferenceEngine.Worker(model, Unity.InferenceEngine.BackendType.CPU);
        }
        if (_pvWorker == null && pvModelAsset != null)
        {
            var pvModel = Unity.InferenceEngine.ModelLoader.Load(pvModelAsset);
            _pvWorker = new Unity.InferenceEngine.Worker(pvModel, Unity.InferenceEngine.BackendType.CPU);
        }
    }

    // ── entry point ─────────────────────────────────────────────────────────

    public void Solve()
    {
        if (IsRunning) return;
        StartCoroutine(SolveCoroutine());
    }

    IEnumerator SolveCoroutine()
    {
        IsRunning = true;
        EnsureWorkers();

        var cube = RubiksCubeController.Instance;
        var ui   = RubikUIController.Instance;

        if (cube.IsSolved())
        {
            ui?.SetStatus("Already solved");
            IsRunning = false;
            yield break;
        }

        var cubies = GetAllCubies();
        CubeStateReader.Encode(cubies, _inputBuffer);

        if (solveMode == SolveMode.Beam)
        {
            bool usingValue = _pvWorker != null;
            ui?.SetStatus(usingValue ? "Beam searching (value-guided)..."
                                     : "Beam searching...");
            yield return null;  // let UI render before blocking main thread

            int[] path = usingValue ? RunBeamSearchPV() : RunBeamSearchPolicyOnly();

            if (path != null && path.Length > 0)
            {
                ui?.SetStatus($"Applying {path.Length} moves...");
                foreach (int idx in path)
                {
                    cube.ApplyMove(MoveTable[idx], addToHistory: false);
                    yield return new WaitUntil(() => !cube.IsAnimating);
                }
            }
        }
        else
        {
            ui?.SetStatus("Neural solving...");
            yield return RunGreedy(cubies, cube, ui);
            IsRunning = false;
            yield break;
        }

        IsRunning = false;

        if (cube.IsSolved())
        {
            ui?.SetStatus("Solved!");
            yield return new WaitForSeconds(2f);
            ui?.SetStatus("");
        }
        else
        {
            ui?.SetStatus("Could not solve");
        }
    }

    // ── greedy ──────────────────────────────────────────────────────────────

    IEnumerator RunGreedy(Cubie[] cubies, RubiksCubeController cube, RubikUIController ui)
    {
        var seenStates = new HashSet<string>();
        int previousIdx = -1;

        for (int step = 0; step < maxGreedySteps; step++)
        {
            if (cube.IsSolved()) break;

            string key = StateKey(_inputBuffer);
            if (seenStates.Contains(key)) { ui?.SetStatus("Loop — stopped"); yield break; }
            seenStates.Add(key);

            InferPolicy(_inputBuffer, _logitsBuffer);
            int best = BestMoveIndex(_logitsBuffer, previousIdx);

            previousIdx = best;
            cube.ApplyMove(MoveTable[best], addToHistory: false);
            yield return new WaitUntil(() => !cube.IsAnimating);

            if (cube.IsSolved()) break;
            CubeStateReader.Encode(cubies, _inputBuffer);
        }

        if (cube.IsSolved())
        {
            ui?.SetStatus("Solved!");
            yield return new WaitForSeconds(2f);
            ui?.SetStatus("");
        }
        else
        {
            ui?.SetStatus("Could not solve");
        }
    }

    // ── beam nodes ──────────────────────────────────────────────────────────

    struct BeamNode
    {
        public float          score;
        public int[]          path;
        public RubikCubeArray cube;
        public int            lastIdx;
    }

    // ── policy-only beam ────────────────────────────────────────────────────

    int[] RunBeamSearchPolicyOnly()
    {
        var initial = RubikCubeArray.FromOneHot(_inputBuffer);
        if (initial.IsSolved()) return Array.Empty<int>();

        var beam = new List<BeamNode>
        {
            new BeamNode { score = 0f, path = Array.Empty<int>(), cube = initial, lastIdx = -1 }
        };

        var enc    = new float[324];
        var logits = new float[12];

        for (int depth = 0; depth < maxBeamDepth; depth++)
        {
            var candidates = new List<BeamNode>(beam.Count * beamTopK);

            foreach (var node in beam)
            {
                node.cube.EncodeOneHot(enc);
                InferPolicy(enc, logits);

                float[] lp = LogSoftmax(logits);

                var order = SortedOrder(logits);
                int forbidden = node.lastIdx >= 0 ? InverseIndex[node.lastIdx] : -1;
                int picked = 0;

                foreach (int idx in order)
                {
                    if (picked >= beamTopK) break;
                    if (idx == forbidden)   continue;

                    var next = node.cube.Clone();
                    next.ApplyMove(idx);

                    var newPath = AppendPath(node.path, idx);
                    if (next.IsSolved()) return newPath;

                    float newScore = node.score - lp[idx];   // −log p, lower = more probable
                    candidates.Add(new BeamNode { score = newScore, path = newPath, cube = next, lastIdx = idx });
                    picked++;
                }
            }

            if (candidates.Count == 0) break;
            candidates.Sort((a, b) => a.score.CompareTo(b.score));
            if (candidates.Count > beamSize) candidates.RemoveRange(beamSize, candidates.Count - beamSize);
            beam = candidates;
        }

        return null;
    }

    // ── value-guided beam ───────────────────────────────────────────────────

    int[] RunBeamSearchPV()
    {
        var initial = RubikCubeArray.FromOneHot(_inputBuffer);
        if (initial.IsSolved()) return Array.Empty<int>();


        var beam = new List<BeamNode>
        {
            new BeamNode { score = 0f, path = Array.Empty<int>(), cube = initial, lastIdx = -1 }
        };

        var enc    = new float[324];
        var logits = new float[12];

        for (int depth = 0; depth < maxBeamDepth; depth++)
        {
            var candidates = new List<BeamNode>(beam.Count * beamTopK);

            foreach (var node in beam)
            {
                node.cube.EncodeOneHot(enc);
                InferPV(enc, logits, out _);

                float[] lp    = LogSoftmax(logits);
                int[]   order = SortedOrder(logits);

                int forbidden = node.lastIdx >= 0 ? InverseIndex[node.lastIdx] : -1;
                int picked    = 0;

                foreach (int idx in order)
                {
                    if (picked >= beamTopK) break;
                    if (idx == forbidden)   continue;

                    var next    = node.cube.Clone();
                    next.ApplyMove(idx);
                    var newPath = AppendPath(node.path, idx);

                    if (next.IsSolved()) return newPath;

                    next.EncodeOneHot(enc);
                    InferPV(enc, logits, out float nextValue);

                    float newScore = pathLengthWeight * newPath.Length
                                   + valueWeight      * nextValue
                                   - policyWeight     * lp[idx];

                    candidates.Add(new BeamNode { score = newScore, path = newPath, cube = next, lastIdx = idx });
                    picked++;
                }
            }

            if (candidates.Count == 0) break;
            candidates.Sort((a, b) => a.score.CompareTo(b.score));
            if (candidates.Count > beamSize) candidates.RemoveRange(beamSize, candidates.Count - beamSize);
            beam = candidates;
        }

        return null;
    }

    // ── inference helpers ────────────────────────────────────────────────────

    void InferPolicy(float[] input, float[] logitsOut)
    {
        using var t = new Unity.InferenceEngine.Tensor<float>(
            new Unity.InferenceEngine.TensorShape(1, 324), input);
        _worker.Schedule(t);
        var raw = _worker.PeekOutput("move_logits") as Unity.InferenceEngine.Tensor<float>;
        raw.CompleteAllPendingOperations();
        for (int i = 0; i < 12; i++) logitsOut[i] = raw[0, i];
    }

    void InferPV(float[] input, float[] logitsOut, out float valueOut)
    {
        using var t = new Unity.InferenceEngine.Tensor<float>(
            new Unity.InferenceEngine.TensorShape(1, 324), input);
        _pvWorker.Schedule(t);
        var rawL = _pvWorker.PeekOutput("move_logits") as Unity.InferenceEngine.Tensor<float>;
        var rawV = _pvWorker.PeekOutput("value")       as Unity.InferenceEngine.Tensor<float>;
        rawL.CompleteAllPendingOperations();
        rawV.CompleteAllPendingOperations();
        for (int i = 0; i < 12; i++) logitsOut[i] = rawL[0, i];
        valueOut = rawV[0, 0];
    }

    // ── math helpers ─────────────────────────────────────────────────────────

    static float[] LogSoftmax(float[] logits)
    {
        float max = float.NegativeInfinity;
        foreach (float v in logits) if (v > max) max = v;
        float sum = 0f;
        var p = new float[logits.Length];
        for (int i = 0; i < logits.Length; i++) { p[i] = MathF.Exp(logits[i] - max); sum += p[i]; }
        for (int i = 0; i < logits.Length; i++) p[i] = MathF.Log(p[i] / sum + 1e-10f);
        return p;
    }

    static int[] SortedOrder(float[] logits)
    {
        var order = new int[logits.Length];
        for (int i = 0; i < logits.Length; i++) order[i] = i;
        Array.Sort(order, (a, b) => logits[b].CompareTo(logits[a]));
        return order;
    }

    static int[] AppendPath(int[] path, int idx)
    {
        var np = new int[path.Length + 1];
        Array.Copy(path, np, path.Length);
        np[path.Length] = idx;
        return np;
    }

    static int BestMoveIndex(float[] logits, int previousIdx)
    {
        int forbidden = previousIdx >= 0 ? InverseIndex[previousIdx] : -1;
        int best = -1;
        float bestVal = float.NegativeInfinity;
        for (int i = 0; i < 12; i++)
        {
            if (i == forbidden) continue;
            if (logits[i] > bestVal) { bestVal = logits[i]; best = i; }
        }
        return best >= 0 ? best : 0;
    }

    // ── utilities ────────────────────────────────────────────────────────────

    static Cubie[] GetAllCubies() =>
        RubiksCubeController.Instance.GetComponentsInChildren<Cubie>();

    static string StateKey(float[] oneHot)
    {
        var sb = new System.Text.StringBuilder(54);
        for (int i = 0; i < 54; i++)
            for (int c = 0; c < 6; c++)
                if (oneHot[i * 6 + c] > 0.5f) { sb.Append((char)('0' + c)); break; }
        return sb.ToString();
    }
}
