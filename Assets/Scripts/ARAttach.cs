using UnityEngine;
public class ARAttach : MonoBehaviour
{
    [Header("Scene References")]
    public GameObject tshirtPrefab;

    [Header("Tracking & Smoothing")]
    public float followSpeed = 10f;

    // This property allows other scripts to get the spawned t-shirt
    public GameObject TshirtInstance { get; private set; }

    private Transform pointListAnnotation;
    private bool isSpawned = false;
    private float originalShoulderWidth;
    private float originalTorsoHeight;
    private Vector3 tshirtLocalShoulderAnchorPoint;
    void LateUpdate()
    {
        if (pointListAnnotation == null)
        {
            GameObject go = GameObject.Find("Point List Annotation");
            if (go != null) { pointListAnnotation = go.transform; }
            else { return; }
        }

        if (!isSpawned)
        {
            InitializeTshirt();
        }

        if (TshirtInstance != null)
        {
            UpdateTshirtTransform();
        }
    }

    private void InitializeTshirt()
    {
        TshirtInstance = Instantiate(tshirtPrefab);
        isSpawned = true;

        Vector3 leftShoulderModelPos = TshirtInstance.transform.Find("LeftShoulder").localPosition;
        Vector3 rightShoulderModelPos = TshirtInstance.transform.Find("RightShoulder").localPosition;
        Vector3 leftHipModelPos = TshirtInstance.transform.Find("LeftHip").localPosition;
        originalShoulderWidth = Vector3.Distance(leftShoulderModelPos, rightShoulderModelPos);
        originalTorsoHeight = Vector3.Distance(leftShoulderModelPos, leftHipModelPos);
        tshirtLocalShoulderAnchorPoint = (leftShoulderModelPos + rightShoulderModelPos) / 2;
    }

    private void UpdateTshirtTransform()
    {
        Vector3 lsh = pointListAnnotation.GetChild(11).position;
        Vector3 rsh = pointListAnnotation.GetChild(12).position;
        Vector3 lhip = pointListAnnotation.GetChild(23).position;
        Vector3 rhip = pointListAnnotation.GetChild(24).position;

        // --- Your Original Position, Rotation, and Scaling Logic ---
        Vector3 topCen = (lsh + rsh) / 2;
        Vector3 offset = Vector3.Scale(TshirtInstance.transform.localScale, tshirtLocalShoulderAnchorPoint);
        Vector3 targetPos = new Vector3(topCen.x - offset.x, topCen.y - offset.y, topCen.z - offset.z - 10);
        TshirtInstance.transform.position = Vector3.Lerp(TshirtInstance.transform.position, targetPos, followSpeed * Time.deltaTime);

        Vector3 shoulderLine = rsh - lsh;
        float angle = Mathf.Atan2(shoulderLine.y, shoulderLine.x) * Mathf.Rad2Deg;
        Quaternion targetRot = Quaternion.Euler(0, 0, angle - 180);
        TshirtInstance.transform.rotation = Quaternion.Slerp(TshirtInstance.transform.rotation, targetRot, followSpeed * Time.deltaTime);

        float detectedWidth = Vector3.Distance(lsh, rsh);
        float detectedHeight = Vector3.Distance(lsh, lhip);
        Vector3 targetScale = new Vector3(detectedWidth / originalShoulderWidth, detectedHeight / originalTorsoHeight, 0.5f);
        TshirtInstance.transform.localScale = Vector3.Lerp(TshirtInstance.transform.localScale, targetScale, followSpeed * Time.deltaTime);
    }
}