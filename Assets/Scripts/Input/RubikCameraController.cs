using System.Collections;
using UnityEngine;

public class RubikCameraController : MonoBehaviour
{
    [SerializeField] private float sensitivity  = 0.3f;
    [SerializeField] private float flipDuration = 0.35f;
    [SerializeField] private float distance     = 9f;
    [SerializeField] private float minDistance  = 5f;
    [SerializeField] private float maxDistance  = 16f;
    [SerializeField] private float zoomSpeed    = 2f;

    private float yaw   = 225f;
    private float pitch =  30f;
    private float roll  =   0f;
    private Vector2 lastMousePos;
    private bool flipping;
    private Transform camTransform;

    void Awake()
    {
        camTransform = GetComponentInChildren<Camera>().transform;
        camTransform.localPosition = new Vector3(0f, 0f, -distance);
    }

    void Update()
    {
        if (flipping) return;

        // Zoom with scroll wheel
        float scroll = Input.GetAxis("Mouse ScrollWheel");
        if (scroll != 0f)
        {
            distance = Mathf.Clamp(distance - scroll * zoomSpeed, minDistance, maxDistance);
            camTransform.localPosition = new Vector3(0f, 0f, -distance);
        }

        if (Input.GetMouseButtonDown(2))
            lastMousePos = Input.mousePosition;

        if (Input.GetMouseButton(2))
        {
            Vector2 current = Input.mousePosition;
            Vector2 delta   = current - lastMousePos;
            lastMousePos    = current;

            float yawDir = Mathf.Abs(roll) > 90f ? -1f : 1f;
            yaw   += delta.x * sensitivity * yawDir;
            pitch -= delta.y * sensitivity;
            pitch  = Mathf.Clamp(pitch, -80f, 80f);

            transform.rotation = Quaternion.Euler(pitch, yaw, roll);
        }
    }

    public void FlipCamera()
    {
        if (!flipping) StartCoroutine(DoFlip());
    }

    private IEnumerator DoFlip()
    {
        flipping = true;
        float fromPitch = pitch;
        float toPitch   = -pitch;
        float fromRoll  = roll;
        float toRoll    = roll == 0f ? 180f : 0f;
        float elapsed   = 0f;

        while (elapsed < flipDuration)
        {
            elapsed += Time.deltaTime;
            float t = Mathf.SmoothStep(0f, 1f, elapsed / flipDuration);
            pitch = Mathf.Lerp(fromPitch, toPitch, t);
            roll  = Mathf.Lerp(fromRoll,  toRoll,  t);
            transform.rotation = Quaternion.Euler(pitch, yaw, roll);
            yield return null;
        }

        pitch = toPitch;
        roll  = toRoll;
        transform.rotation = Quaternion.Euler(pitch, yaw, roll);
        flipping = false;
    }
}
