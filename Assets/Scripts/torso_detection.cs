using UnityEngine;
using System.Collections.Generic;
using Unity.Barracuda;
using UnityEngine.UI;

public class torso_detection : MonoBehaviour
{
    [Header("Model Settings")]
    [SerializeField] private NNModel modelAsset;
    [SerializeField] private WebCamTexture webCamTexture;
    [SerializeField] private RawImage displayImage;

    [Header("Clothing Settings")]
    [SerializeField] private GameObject clothingPrefab;
    [SerializeField] private Transform torsoAnchor;

    private Vector3[] torsoKeypoints = new Vector3[8];
    private Model runtimeModel;
    private IWorker worker;
    private GameObject currentClothing;
    private RenderTexture rt;

    private void Start()
    {
        InitializeCamera();
        InitializeModel();
        InitializeClothing();
    }

    private void InitializeCamera()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.LogError("No camera found");
            return;
        }
        webCamTexture = new WebCamTexture(devices[0].name, 1280, 720, 30);
        webCamTexture.Play();
        displayImage.texture = webCamTexture;
        rt = new RenderTexture(1280, 720, 0, RenderTextureFormat.ARGB32);
    }

    private void InitializeModel()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);
    }

    private void InitializeClothing()
    {
        if (clothingPrefab != null)
        {
            currentClothing = Instantiate(clothingPrefab, torsoAnchor);
            currentClothing.SetActive(false);
        }
    }

    private void Update()
    {
        if (!webCamTexture.isPlaying) return;

        DetectTorso();

        if (IsTorsoDetected())
        {
            UpdateClothing();
        }
    }

    private void DetectTorso()
    {
        Graphics.Blit(webCamTexture, rt);
        using (var inputTensor = new Tensor(rt, channels: 3))
        {
            for (int b = 0; b < inputTensor.batch; b++)
            {
                for (int h = 0; h < inputTensor.height; h++)
                {
                    for (int w = 0; w < inputTensor.width; w++)
                    {
                        for (int c = 0; c < inputTensor.channels; c++)
                        {
                            int index = inputTensor.Index(b, h, w, c);
                            inputTensor[index] = (inputTensor[index] / 255.0f) * 2.0f - 1.0f;
                        }
                    }
                }
            }
            worker.Execute(inputTensor);

            var output = worker.PeekOutput();

            ProcessKeypointsFromTensor(output);
        }
    }

    private void ProcessKeypointsFromTensor(Tensor output)
    {
        for (int i = 0; i < 8; i++)
        {
            float x = output[i * 3];
            float y = output[i * 3 + 1];
            float confidence = output[i * 3 + 2];

            if (confidence > 0.5f)
            {
                Debug.Log("working?");
                torsoKeypoints[i] = Camera.main.ViewportToWorldPoint(
                    new Vector3(x, y, Camera.main.nearClipPlane));
            }
            else
            {
                Debug.Log("confidence: " + confidence + " not working ;(");
            }
        }
    }

    private bool IsTorsoDetected()
    {
        int validKeypoints = 0;
        foreach (var keypoint in torsoKeypoints)
        {
            if (keypoint != Vector3.zero) validKeypoints++;
        }
        return validKeypoints >= 6;
    }

    private void UpdateClothing()
    {
        if (currentClothing == null) return;

        if (!currentClothing.activeSelf)
            currentClothing.SetActive(true);

        Vector3 torsoCenter = CalculateTorsoCenter();
        torsoAnchor.position = torsoCenter;

        Vector3 shoulderLine = torsoKeypoints[1] - torsoKeypoints[0];
        float angle = Mathf.Atan2(shoulderLine.y, shoulderLine.x) * Mathf.Rad2Deg;
        torsoAnchor.rotation = Quaternion.Euler(0, 0, angle);

        float torsoWidth = Vector3.Distance(torsoKeypoints[0], torsoKeypoints[1]);
        float torsoHeight = Vector3.Distance(
            (torsoKeypoints[0] + torsoKeypoints[1]) * 0.5f,
            (torsoKeypoints[6] + torsoKeypoints[7]) * 0.5f);
        torsoAnchor.localScale = new Vector3(torsoWidth, torsoHeight, 1);

        UpdateClothPhysics();
    }

    private Vector3 CalculateTorsoCenter()
    {
        Vector3 sum = Vector3.zero;
        foreach (var keypoint in torsoKeypoints)
        {
            sum += keypoint;
        }
        return sum / torsoKeypoints.Length;
    }

    private void UpdateClothPhysics()
    {
        if (currentClothing.TryGetComponent<Cloth>(out var cloth))
        {
            var coefficients = new ClothSkinningCoefficient[torsoKeypoints.Length];
            for (int i = 0; i < torsoKeypoints.Length; i++)
            {
                coefficients[i] = new ClothSkinningCoefficient
                {
                    maxDistance = 0.1f,
                    collisionSphereDistance = 0.05f
                };
            }
            cloth.coefficients = coefficients;
        }
    }

    private void OnDestroy()
    {
        worker?.Dispose();
        if (webCamTexture != null && webCamTexture.isPlaying)
            webCamTexture.Stop();
    }
}