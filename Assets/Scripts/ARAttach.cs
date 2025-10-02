using UnityEngine;
public class ARAttach : MonoBehaviour
{
    [Header("Scene References")]
    public GameObject tshirtPrefab;

    [Header("Tracking & Smoothing")]
    public float followSpeed = 10f;

    public GameObject TshirtInstance { get; private set; }

    private Transform pointListAnnotation;
    private bool isSpawned = false;
    private float originalShoulderWidth;
    private float originalTorsoHeight;
    private Vector3 tshirtLocalShoulderAnchorPoint;

    private Transform spineBone;
    private Transform leftUpperArmBone;
    private Transform rightUpperArmBone;
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

        spineBone = TshirtInstance.transform.Find("bones/Armature/Bone/Spine");
        leftUpperArmBone = TshirtInstance.transform.Find("bones/Armature/Bone/Spine/Shoulder_L/Arm_L");
        rightUpperArmBone = TshirtInstance.transform.Find("bones/Armature/Bone/Spine/Shoulder_R/Arm_R");
        var bounds = new Bounds();
        bounds.center = TshirtInstance.transform.Find("LeftShoulder").localPosition;
        bounds.Encapsulate(TshirtInstance.transform.Find("RightShoulder").localPosition);
        bounds.Encapsulate(TshirtInstance.transform.Find("LeftHip").localPosition);
        bounds.Encapsulate(TshirtInstance.transform.Find("RightHip").localPosition);

        originalShoulderWidth = bounds.size.x;
        originalTorsoHeight = bounds.size.y;

        Vector3 leftShoulderModelPos = TshirtInstance.transform.Find("LeftShoulder").localPosition;
        Vector3 rightShoulderModelPos = TshirtInstance.transform.Find("RightShoulder").localPosition;
        tshirtLocalShoulderAnchorPoint = (leftShoulderModelPos + rightShoulderModelPos) / 2;

        Debug.Log("T-shirt initialized and placed.");

    }

    private void UpdateTshirtTransform()
    {
        Vector3 lsh = pointListAnnotation.GetChild(11).position; // Left Shoulder
        Vector3 rsh = pointListAnnotation.GetChild(12).position; // Right Shoulder
        Vector3 lel = pointListAnnotation.GetChild(13).position; // Left Elbow
        Vector3 rel = pointListAnnotation.GetChild(14).position; // Right Elbow
        Vector3 lhip = pointListAnnotation.GetChild(23).position; // Left Hip
        Vector3 rhip = pointListAnnotation.GetChild(24).position; // Right Hip

        Vector3 shoulderCenter = (lsh + rsh) / 2;
        Vector3 hipCenter = (lhip + rhip) / 2;
        Vector3 torsoUp = (shoulderCenter - hipCenter).normalized;
        Vector3 torsoRight = (rsh - lsh).normalized;
        Vector3 torsoForward = Vector3.Cross(torsoUp, torsoRight).normalized;
        Quaternion targetRot = Quaternion.LookRotation(torsoForward, torsoUp);

        float detectedTorsoHeight = Vector3.Distance(shoulderCenter, hipCenter);
        float scaleFactor = detectedTorsoHeight / originalTorsoHeight;
        Vector3 targetScale = Vector3.one * scaleFactor;

        Vector3 scaledAndRotatedAnchorOffset = targetRot * (tshirtLocalShoulderAnchorPoint * scaleFactor);
        Vector3 targetPos = shoulderCenter - scaledAndRotatedAnchorOffset;

        float speed = followSpeed * Time.deltaTime;
        TshirtInstance.transform.position = Vector3.Lerp(TshirtInstance.transform.position, targetPos, speed);
        TshirtInstance.transform.rotation = Quaternion.Slerp(TshirtInstance.transform.rotation, targetRot, speed);
        TshirtInstance.transform.localScale = Vector3.Lerp(TshirtInstance.transform.localScale, targetScale, speed);
        /*if (leftUpperArmBone != null)
        {
            Vector3 leftArmDir = (lel - lsh).normalized;
            Quaternion leftArmRot = Quaternion.FromToRotation(Vector3.left, leftArmDir);
            Quaternion finalLeftArmRotation = targetRot * leftArmRot;

            leftUpperArmBone.rotation = Quaternion.Slerp(leftUpperArmBone.rotation, finalLeftArmRotation, speed);
        }
        if (rightUpperArmBone != null)
        {
            Vector3 rightArmDir = (rel - rsh).normalized;
            Quaternion rightArmRot = Quaternion.FromToRotation(Vector3.right, rightArmDir);
            Quaternion finalRightRot = targetRot * rightArmRot;
            rightUpperArmBone.rotation = Quaternion.Slerp(rightUpperArmBone.rotation, finalRightRot, speed);
        }*/
    }
}