using UnityEngine;

public class RubikInputController : MonoBehaviour
{
    [SerializeField] private Camera mainCamera;
    [SerializeField] private float dragThreshold = 10f;

    private Vector2 dragStartMouse;
    private Vector3Int clickedCoord;
    private Vector3Int clickedNormal;
    private bool waitingForDrag = false;

    void Update()
    {
        if (RubiksCubeController.Instance == null || RubiksCubeController.Instance.IsAnimating) return;

        if (Input.GetMouseButtonDown(0))
        {
            TryBeginDrag();
        }
        else if (Input.GetMouseButtonUp(0) && waitingForDrag)
        {
            TryExecuteMove();
            waitingForDrag = false;
        }
    }

    void TryBeginDrag()
    {
        Ray ray = mainCamera.ScreenPointToRay(Input.mousePosition);
        if (Physics.Raycast(ray, out RaycastHit hit))
        {
            StickerFace sticker = hit.collider.GetComponentInParent<StickerFace>();
            if (sticker != null && Vector3.Dot(sticker.transform.forward, hit.point - mainCamera.transform.position) < 0)
            {
                dragStartMouse = Input.mousePosition;
                clickedCoord = sticker.Cubie.Coord;
                // LocalNormal is stale after layer turns — derive from current world transform
                clickedNormal = WorldNormalToAxis(sticker.transform.forward);
                waitingForDrag = true;
            }
        }
    }

    static Vector3Int WorldNormalToAxis(Vector3 v)
    {
        float ax = Mathf.Abs(v.x), ay = Mathf.Abs(v.y), az = Mathf.Abs(v.z);
        if (ax >= ay && ax >= az) return new Vector3Int((int)Mathf.Sign(v.x), 0, 0);
        if (ay >= ax && ay >= az) return new Vector3Int(0, (int)Mathf.Sign(v.y), 0);
        return new Vector3Int(0, 0, (int)Mathf.Sign(v.z));
    }

    void TryExecuteMove()
    {
        Vector2 current = Input.mousePosition;
        RubikMove? move = TryCreateMove(clickedCoord, clickedNormal, dragStartMouse, current, mainCamera);
        if (move.HasValue)
        {
            waitingForDrag = false;
            RubiksCubeController.Instance.ApplyMove(move.Value, addToHistory: true);
        }
    }

    RubikMove? TryCreateMove(Vector3Int coord, Vector3Int normal, Vector2 mouseStart, Vector2 mouseCurrent, Camera cam)
    {
        Vector2 drag = mouseCurrent - mouseStart;
        if (drag.magnitude < dragThreshold)
            return null;

        Vector3Int tangentA, tangentB;
        if (Mathf.Abs(normal.x) == 1)
        {
            tangentA = Vector3Int.up;
            tangentB = new Vector3Int(0, 0, 1);
        }
        else if (Mathf.Abs(normal.y) == 1)
        {
            tangentA = Vector3Int.right;
            tangentB = new Vector3Int(0, 0, 1);
        }
        else
        {
            tangentA = Vector3Int.right;
            tangentB = Vector3Int.up;
        }

        Vector2 screenA = ProjectToScreen(cam, tangentA);
        Vector2 screenB = ProjectToScreen(cam, tangentB);

        float dotA = Mathf.Abs(Vector2.Dot(drag.normalized, screenA.normalized));
        float dotB = Mathf.Abs(Vector2.Dot(drag.normalized, screenB.normalized));

        Vector3Int tangent = dotA >= dotB ? tangentA : tangentB;
        Vector2 chosenScreen = dotA >= dotB ? screenA : screenB;

        int tangentSign = Vector2.Dot(drag, chosenScreen) >= 0 ? 1 : -1;
        Vector3Int signedTangent = tangent * tangentSign;
        Vector3Int signedAxis = RubiksCubeController.Cross(normal, signedTangent);

        RubikAxis axis = RubiksCubeController.AbsoluteAxisOf(signedAxis);
        int layer = RubiksCubeController.GetCoordOnAxis(coord, axis);
        int direction = RubiksCubeController.SignAlongAxis(signedAxis, axis);

        if (!RubiksCubeController.Instance.allowMiddleSlices && layer == 0)
            return null;

        return new RubikMove(axis, layer, direction);
    }

    Vector2 ProjectToScreen(Camera cam, Vector3Int cubeLocalDir)
    {
        Vector3 origin = cam.WorldToScreenPoint(Vector3.zero);
        Vector3 tip = cam.WorldToScreenPoint((Vector3)(Vector3)cubeLocalDir);
        return new Vector2(tip.x - origin.x, tip.y - origin.y);
    }
}
