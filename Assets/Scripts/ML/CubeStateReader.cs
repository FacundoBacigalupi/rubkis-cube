using UnityEngine;

/// <summary>
/// Reads the current cube state from live GameObjects and produces the
/// float[324] one-hot tensor that the ONNX model expects.
///
/// Python sticker layout (must match rubikml/cube.py exactly):
///   face * 9 + row * 3 + col
///   U=0  D=1  R=2  L=3  F=4  B=5
///
/// Color ids:  U=0  D=1  R=2  L=3  F=4  B=5
/// One-hot:    sticker i with color c  →  float[i*6 + c] = 1
/// </summary>
public static class CubeStateReader
{
    // Maps a normal direction → Python color id (sticker's fixed original color)
    static int NormalToColorId(Vector3Int n)
    {
        if (n.y ==  1) return 0; // U
        if (n.y == -1) return 1; // D
        if (n.x ==  1) return 2; // R
        if (n.x == -1) return 3; // L
        if (n.z ==  1) return 4; // F
        if (n.z == -1) return 5; // B
        return -1;
    }

    // Snaps a continuous forward vector to the nearest axis
    static Vector3Int SnapNormal(Vector3 v)
    {
        float ax = Mathf.Abs(v.x), ay = Mathf.Abs(v.y), az = Mathf.Abs(v.z);
        if (ax >= ay && ax >= az) return new Vector3Int((int)Mathf.Sign(v.x), 0, 0);
        if (ay >= ax && ay >= az) return new Vector3Int(0, (int)Mathf.Sign(v.y), 0);
        return new Vector3Int(0, 0, (int)Mathf.Sign(v.z));
    }

    /// <summary>
    /// Returns the sticker index (0-53) in the Python flat array for a face
    /// identified by its current outward normal, on the cubie at logical Coord.
    ///
    /// Row/col conventions match the Python simulator's _f(face, row, col):
    ///
    ///   U (+Y): row = 1+z  (z=-1→row0, z=0→row1, z=1→row2)
    ///           col = 1+x  (x=-1→col0, x=0→col1, x=1→col2)
    ///
    ///   D (-Y): row = 1-z  (z=1→row0, z=0→row1, z=-1→row2)
    ///           col = 1+x
    ///
    ///   R (+X): row = 1-y  (y=1→row0, y=0→row1, y=-1→row2)
    ///           col = 1-z  (z=1→col0, z=0→col1, z=-1→col2)
    ///
    ///   L (-X): row = 1-y
    ///           col = 1+z  (z=-1→col0, z=0→col1, z=1→col2)
    ///
    ///   F (+Z): row = 1-y
    ///           col = 1+x
    ///
    ///   B (-Z): row = 1-y
    ///           col = 1-x  (x=1→col0, x=0→col1, x=-1→col2)
    /// </summary>
    static int StickerIndex(Vector3Int currentNormal, Vector3Int coord)
    {
        int face, row, col;
        int x = coord.x, y = coord.y, z = coord.z;

        if (currentNormal.y == 1)       { face = 0; row = 1 + z; col = 1 + x; }
        else if (currentNormal.y == -1) { face = 1; row = 1 - z; col = 1 + x; }
        else if (currentNormal.x == 1)  { face = 2; row = 1 - y; col = 1 - z; }
        else if (currentNormal.x == -1) { face = 3; row = 1 - y; col = 1 + z; }
        else if (currentNormal.z == 1)  { face = 4; row = 1 - y; col = 1 + x; }
        else                            { face = 5; row = 1 - y; col = 1 - x; }

        return face * 9 + row * 3 + col;
    }

    /// <summary>
    /// Fills a pre-allocated float[324] with the one-hot encoded cube state.
    /// </summary>
    public static void Encode(Cubie[] cubies, float[] output)
    {
        System.Array.Clear(output, 0, 324);

        foreach (var cubie in cubies)
        {
            foreach (var sf in cubie.GetComponentsInChildren<StickerFace>(true))
            {
                if (!sf.gameObject.activeSelf) continue;

                // currentNormal: which face this sticker is currently on (determines array slot)
                // LocalNormal:   which face this sticker originally belonged to (its fixed color)
                Vector3Int currentNormal = SnapNormal(sf.transform.forward);
                int colorId    = NormalToColorId(sf.LocalNormal);
                int stickerIdx = StickerIndex(currentNormal, cubie.Coord);

                if (colorId < 0 || stickerIdx < 0 || stickerIdx >= 54) continue;

                output[stickerIdx * 6 + colorId] = 1f;
            }
        }
    }
}
