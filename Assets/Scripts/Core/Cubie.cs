using UnityEngine;

public class Cubie : MonoBehaviour
{
    public Vector3Int Coord { get; private set; }

    private const float Spacing = 1.02f;

    public void SetCoord(Vector3Int coord)
    {
        Coord = coord;
        transform.localPosition = new Vector3(coord.x, coord.y, coord.z) * Spacing;
    }

    public void UpdateCoord(Vector3Int coord)
    {
        Coord = coord;
    }
}
