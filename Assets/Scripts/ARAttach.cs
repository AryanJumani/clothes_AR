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

    [Header("Rig Bones")]
    public Transform spineBone;
    public Transform leftUpperArmBone;
    public Transform rightUpperArmBone;
    void LateUpdate()
    {
        if (pointListAnnotation == null)
        {
            GameObject go = GameObject.Find("Point List Annotation");
            if (go != null) { pointListAnnotation = go.transform; }
            else
            {
                if (TshirtInstance != null)
                {
                    Destroy(TshirtInstance);
                    isSpawned = false;
                }

                return;
            }
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

        var bounds = new Bounds();
        bounds.center = TshirtInstance.transform.Find("LeftShoulder").localPosition;
        bounds.Encapsulate(TshirtInstance.transform.Find("RightShoulder").localPosition);
        bounds.Encapsulate(TshirtInstance.transform.Find("LeftHip").localPosition);
        bounds.Encapsulate(TshirtInstance.transform.Find("RightHip").localPosition);

        originalShoulderWidth = bounds.size.x;
        originalTorsoHeight = bounds.size.y;

        // The anchor point is the center of the shoulder line in the model's local space
        Vector3 leftShoulderModelPos = TshirtInstance.transform.Find("LeftShoulder").localPosition;
        Vector3 rightShoulderModelPos = TshirtInstance.transform.Find("RightShoulder").localPosition;
        tshirtLocalShoulderAnchorPoint = (leftShoulderModelPos + rightShoulderModelPos) / 2;

        Debug.Log("T-shirt initialized and placed.");

    }

    private void UpdateTshirtTransform()
    {
        // Get the world positions of the 4 key landmarks
        Vector3 lsh = pointListAnnotation.GetChild(11).position; // Left Shoulder
        Vector3 rsh = pointListAnnotation.GetChild(12).position; // Right Shoulder
        Vector3 lel = pointListAnnotation.GetChild(13).position;
        Vector3 rel = pointListAnnotation.GetChild(14).position;
        Vector3 lhip = pointListAnnotation.GetChild(23).position; // Left Hip
        Vector3 rhip = pointListAnnotation.GetChild(24).position; // Right Hip

        // --- 1. Robust 3D Rotation (No changes here) ---
        Vector3 shoulderCenter = (lsh + rsh) / 2;
        Vector3 hipCenter = (lhip + rhip) / 2;
        Vector3 torsoUp = (shoulderCenter - hipCenter).normalized;
        Vector3 torsoRight = (rsh - lsh).normalized;
        Vector3 torsoForward = Vector3.Cross(torsoUp, torsoRight).normalized;
        Quaternion targetRot = Quaternion.LookRotation(torsoForward, torsoUp);

        // --- 2. Stable Uniform Scale (No changes here) ---
        float detectedTorsoHeight = Vector3.Distance(shoulderCenter, hipCenter);
        float scaleFactor = detectedTorsoHeight / originalTorsoHeight;
        Vector3 targetScale = Vector3.one * scaleFactor;

        // --- 3. POSITIONING FIX ---
        // The previous logic used the center of the whole torso.
        // This new logic directly aligns the T-shirt's shoulder anchor
        // with the detected center of your shoulders. This should fix the "too down" issue.
        Vector3 scaledAndRotatedAnchorOffset = targetRot * (tshirtLocalShoulderAnchorPoint * scaleFactor);
        Vector3 targetPos = shoulderCenter - scaledAndRotatedAnchorOffset;

        // --- 4. Apply Transformations with Smoothing ---
        float speed = followSpeed * Time.deltaTime;
        TshirtInstance.transform.position = Vector3.Lerp(TshirtInstance.transform.position, targetPos, speed);
        TshirtInstance.transform.rotation = Quaternion.Slerp(TshirtInstance.transform.rotation, targetRot, speed);
        TshirtInstance.transform.localScale = Vector3.Lerp(TshirtInstance.transform.localScale, targetScale, speed);

        if (spineBone != null)
        {
            Vector3 spineDir = shoulderCenter - hipCenter;
            Quaternion spineRot = Quaternion.LookRotation(spineDir, torsoRight);
            spineBone.rotation = Quaternion.Slerp(spineBone.rotation, spineRot, speed);
        }
        if (leftUpperArmBone != null)
        {
            Vector3 leftArmDir = lel - lsh;
            Quaternion leftArmRot = Quaternion.LookRotation(leftArmDir, torsoUp);
            leftUpperArmBone.rotation = Quaternion.Slerp(leftUpperArmBone.rotation, leftArmRot, speed);
        }
        if (rightUpperArmBone != null)
        {
            Vector3 rightArmDir = rel - rsh;
            Quaternion rightArmRot = Quaternion.LookRotation(rightArmDir, torsoUp);
            rightUpperArmBone.rotation = Quaternion.Slerp(rightUpperArmBone.rotation, rightArmRot, speed);
        }

    }

}