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

    [Header("Debug")]
    [SerializeField] private bool showDebugInfo = true;

    private readonly Vector3[] torsoKeypoints = new Vector3[8];
    private Model runtimeModel;
    private IWorker worker;
    private GameObject currentClothing;
    private Texture2D processingTexture;
    private bool isInitialized = false;
    private float[] preprocessedData;

    private void Start()
    {
        InitializeCamera();
    }

    private void InitializeCamera()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length == 0)
        {
            Debug.LogError("No camera found");
            return;
        }

        webCamTexture = new WebCamTexture(devices[0].name, 640, 480, 30);
        webCamTexture.Play();

        StartCoroutine(WaitForWebcam());
    }

    private System.Collections.IEnumerator WaitForWebcam()
    {
        while (webCamTexture.width <= 16)
        {
            yield return null;
        }

        Debug.Log($"WebCam initialized with dimensions: {webCamTexture.width}x{webCamTexture.height}");
        processingTexture = new Texture2D(webCamTexture.width, webCamTexture.height, TextureFormat.RGB24, false);
        displayImage.texture = webCamTexture;

        // Preallocate array for preprocessed data
        preprocessedData = new float[webCamTexture.width * webCamTexture.height * 3];

        InitializeModel();
        InitializeClothing();

        isInitialized = true;
    }

    private void InitializeModel()
    {
        runtimeModel = ModelLoader.Load(modelAsset);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);

        if (showDebugInfo)
        {
            Debug.Log($"Model inputs: {string.Join(", ", runtimeModel.inputs)}");
            Debug.Log($"Model outputs: {string.Join(", ", runtimeModel.outputs)}");
        }
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
        if (!isInitialized || !webCamTexture.isPlaying) return;

        DetectTorso();

        if (IsTorsoDetected())
        {
            UpdateClothing();
        }
    }

    private void DetectTorso()
    {
        try
        {
            Color32[] pixels = webCamTexture.GetPixels32();

            // Preprocess the image data
            PreprocessPixels(pixels);

            // Create tensor from preprocessed data
            var tensorShape = new TensorShape(1, webCamTexture.height, webCamTexture.width, 3);
            using (var tensor = new Tensor(tensorShape, preprocessedData))
            {
                worker.Execute(tensor);

                // Process all outputs
                foreach (string outputName in runtimeModel.outputs)
                {
                    using (var output = worker.PeekOutput(outputName))
                    {
                        if (showDebugInfo)
                        {
                            Debug.Log($"Output {outputName} shape: {output.shape}");
                            Debug.Log($"First few values of {outputName}: {output[0]}, {output[1]}, {output[2]}");
                        }
                    }
                }

                // Get the main output for keypoints
                using (var output = worker.PeekOutput(runtimeModel.outputs[0]))
                {
                    ProcessKeypointsFromTensor(output);
                }
            }
        }
        catch (System.Exception e)
        {
            Debug.LogError($"Error processing frame: {e.Message}\nStack trace: {e.StackTrace}");
        }
    }

    private void PreprocessPixels(Color32[] pixels)
    {
        for (int i = 0; i < pixels.Length; i++)
        {
            // Convert to float and normalize to [-1, 1]
            preprocessedData[i * 3] = (pixels[i].r / 255.0f) * 2.0f - 1.0f;     // R
            preprocessedData[i * 3 + 1] = (pixels[i].g / 255.0f) * 2.0f - 1.0f; // G
            preprocessedData[i * 3 + 2] = (pixels[i].b / 255.0f) * 2.0f - 1.0f; // B
        }
    }

    private void ProcessKeypointsFromTensor(Tensor output)
    {
        if (showDebugInfo)
        {
            Debug.Log($"Processing output tensor with shape: {output.shape}");
        }

        // Assuming the first output contains keypoint data
        for (int i = 0; i < 8; i++)
        {
            float x = output[i * 3];
            float y = output[i * 3 + 1];
            float confidence = output[i * 3 + 2];

            if (showDebugInfo && i == 0)
            {
                Debug.Log($"Keypoint 0 - Raw values - x: {x}, y: {y}, confidence: {confidence}");
            }

            // Convert to screen coordinates
            x = (x + 1f) * 0.5f;
            y = (y + 1f) * 0.5f;
            confidence = Sigmoid(confidence);

            if (showDebugInfo && i == 0)
            {
                Debug.Log($"Keypoint 0 - Processed values - x: {x}, y: {y}, confidence: {confidence}");
            }

            if (confidence > 0.5f)
            {
                torsoKeypoints[i] = Camera.main.ViewportToWorldPoint(
                    new Vector3(x, y, Camera.main.nearClipPlane));
            }
        }
    }

    private float Sigmoid(float x)
    {
        return 1f / (1f + Mathf.Exp(-x));
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
        if (processingTexture != null)
            Destroy(processingTexture);
    }
}