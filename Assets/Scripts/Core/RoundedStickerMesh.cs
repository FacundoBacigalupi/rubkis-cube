using System.Collections.Generic;
using UnityEngine;

[RequireComponent(typeof(MeshFilter))]
public class RoundedStickerMesh : MonoBehaviour
{
    [SerializeField, Range(0.01f, 0.45f)] float cornerRadius = 0.18f;
    [SerializeField, Range(3, 16)]        int   cornerSegments = 6;

    void Awake() => GetComponent<MeshFilter>().mesh = Build(cornerRadius, cornerSegments);

    static Mesh Build(float r, int segs)
    {
        const float h = 0.5f;
        r = Mathf.Min(r, h - 0.001f);

        Vector2[] centers  = { new(-h+r,-h+r), new(h-r,-h+r), new(h-r,h-r), new(-h+r,h-r) };
        float[]   startDeg = { 180f, 270f, 0f, 90f };

        var verts = new List<Vector3> { Vector3.zero };
        for (int c = 0; c < 4; c++)
            for (int i = 0; i <= segs; i++)
            {
                float a = (startDeg[c] + i * 90f / segs) * Mathf.Deg2Rad;
                verts.Add(new Vector3(centers[c].x + r * Mathf.Cos(a),
                                     centers[c].y + r * Mathf.Sin(a), 0f));
            }

        int total = 4 * (segs + 1);
        var tris = new List<int>();
        for (int i = 0; i < total; i++)
        {
            tris.Add(0);
            tris.Add(1 + i);
            tris.Add(1 + (i + 1) % total);
        }

        var uvs = new List<Vector2> { new(0.5f, 0.5f) };
        for (int i = 1; i < verts.Count; i++)
            uvs.Add(new Vector2(verts[i].x + 0.5f, verts[i].y + 0.5f));

        var mesh = new Mesh { name = "RoundedSticker" };
        mesh.SetVertices(verts);
        mesh.SetTriangles(tris, 0);
        mesh.SetUVs(0, uvs);
        mesh.RecalculateNormals();
        return mesh;
    }
}
