using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class LayerTurnAnimator : MonoBehaviour
{
    [SerializeField] private float turnDuration = 0.25f;

    public void AnimateMove(RubikMove move, bool addToHistory, Action<RubikMove> onComplete)
    {
        StartCoroutine(DoAnimateMove(move, addToHistory, onComplete));
    }

    private IEnumerator DoAnimateMove(RubikMove move, bool addToHistory, Action<RubikMove> onComplete)
    {
        var cube = RubiksCubeController.Instance;
        var layerCubies = cube.GetCubiesOnLayer(move.Axis, move.Layer);

        GameObject pivot = new GameObject("LayerPivot");
        pivot.transform.position = Vector3.zero;

        foreach (var c in layerCubies)
            c.transform.SetParent(pivot.transform, true);

        Vector3 axisVec = move.Axis switch
        {
            RubikAxis.X => Vector3.right,
            RubikAxis.Y => Vector3.up,
            RubikAxis.Z => Vector3.forward,
            _ => Vector3.up
        };

        float targetAngle = 90f * move.Direction;
        float elapsed = 0f;
        Quaternion startRot = pivot.transform.rotation;
        Quaternion endRot = Quaternion.AngleAxis(targetAngle, axisVec) * startRot;

        while (elapsed < turnDuration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.Clamp01(elapsed / turnDuration);
            pivot.transform.rotation = Quaternion.Slerp(startRot, endRot, t);
            yield return null;
        }

        pivot.transform.rotation = endRot;

        foreach (var c in layerCubies)
        {
            c.transform.SetParent(cube.transform, true);
            SnapTransform(c.transform);
        }

        Destroy(pivot);

        if (addToHistory)
            RubikMoveHistory.Instance.Push(move);

        onComplete?.Invoke(move);
    }

    private const float Spacing = 1.02f;

    private void SnapTransform(Transform t)
    {
        Vector3 p = t.localPosition;
        t.localPosition = new Vector3(
            Mathf.Round(p.x / Spacing) * Spacing,
            Mathf.Round(p.y / Spacing) * Spacing,
            Mathf.Round(p.z / Spacing) * Spacing);

        Vector3 e = t.localEulerAngles;
        t.localEulerAngles = new Vector3(SnapAngle(e.x), SnapAngle(e.y), SnapAngle(e.z));
    }

    private float SnapAngle(float angle)
    {
        return Mathf.Round(angle / 90f) * 90f;
    }
}
