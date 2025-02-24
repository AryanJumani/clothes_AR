using UnityEngine;
using Unity.Barracuda;

public class PoseInterface : MonoBehaviour
{
    public NNModel pose;
    private Model runtimeModel;
    private IWorker worker;
    private WebcamInput webcamInput;

    void Start()
    {
        runtimeModel = ModelLoader.Load(pose);
        worker = WorkerFactory.CreateWorker(WorkerFactory.Type.ComputePrecompiled, runtimeModel);
        webcamInput = FindObjectOfType<WebcamInput>();
    }

    void Update()
    {
        Texture2D frame = webcamInput.GetFrame();
        if (frame != null)
        {
            Tensor input = PreprocessFrame(frame);
            Tensor output = RunInference(input);
            ExtractKeypoints(output);
            input.Dispose();
            output.Dispose();
        }
    }

    Tensor PreprocessFrame(Texture2D frame)
    {
        int size = 256;
        Texture2D resized = ResizeTexture(frame, size, size);
        float[] data = new float[size * size * 3];
        Color[] pixels = resized.GetPixels();

        for (int i = 0; i < pixels.Length; i++)
        {
            data[i * 3] = pixels[i].r;  // Normalize to [0,1] (MediaPipe expects this)
            data[i * 3 + 1] = pixels[i].g;
            data[i * 3 + 2] = pixels[i].b;
        }

        return new Tensor(1, size, size, 3, data);
    }


    Texture2D ResizeTexture(Texture2D source, int width, int height)
    {
        RenderTexture rt = new RenderTexture(width, height, 24);
        RenderTexture.active = rt;
        Graphics.Blit(source, rt);
        Texture2D result = new Texture2D(width, height);
        result.ReadPixels(new Rect(0, 0, width, height), 0, 0);
        result.Apply();
        return result;
    }

    public Tensor RunInference(Tensor input)
    {
        worker.Execute(input);
        return worker.PeekOutput();
    }

    void OnDestroy()
    {
        worker.Dispose();
    }

    Vector2 NormalizeKeypoint(float x, float y, int width, int height)
    {
        return new Vector2(x * width, y * height); // Scale to pixel space
    }


    void ExtractKeypoints(Tensor output)
    {
        if (output.length < 33 * 4) return; // Ensure expected output size

        int keypointCount = 33;
        float minConfidence = 0.3f; // Confidence threshold

        Debug.Log("Raw Keypoint Outputs:");
        for (int i = 0; i < keypointCount; i++)
        {
            float x = output[i * 4];      // X position (normalized 0-1)
            float y = output[i * 4 + 1];  // Y position (normalized 0-1)
            float z = output[i * 4 + 2];  // Z position (depth)
            float confidence = output[i * 4 + 3]; // Confidence score (0-1)

            Debug.Log($"Keypoint {i}: X={x}, Y={y}, Z={z}, Confidence={confidence}");
        }

        // **Detect if no person is present**
        bool personDetected = false;
        for (int i = 0; i < keypointCount; i++)
        {
            float confidence = output[i * 4 + 3]; // Correct confidence index for MediaPipe
            if (confidence >= minConfidence)
            {
                personDetected = true;
                break;
            }
        }

        if (!personDetected)
        {
            Debug.Log("No person detected! Skipping keypoints.");
            return;
        }

        Debug.LogWarning("Person detected. Extracting keypoints...");

        int width = 256;
        int height = 256;

        Vector2 leftShoulder = NormalizeKeypoint(output[11 * 4], output[11 * 4 + 1], width, height);
        Vector2 rightShoulder = NormalizeKeypoint(output[12 * 4], output[12 * 4 + 1], width, height);
        Vector2 leftHip = NormalizeKeypoint(output[23 * 4], output[23 * 4 + 1], width, height);
        Vector2 rightHip = NormalizeKeypoint(output[24 * 4], output[24 * 4 + 1], width, height);

        Debug.Log($"Left Shoulder: {leftShoulder} | Right Shoulder: {rightShoulder} | Left Hip: {leftHip} | Right Hip: {rightHip}");

        DrawTorsoBox(leftShoulder, rightShoulder, leftHip, rightHip);
    }


    void DrawTorsoBox(Vector2 leftShoulder, Vector2 rightShoulder, Vector2 leftHip, Vector2 rightHip)
    {
        Vector2 topLeft = new Vector2(Mathf.Min(leftShoulder.x, rightShoulder.x), Mathf.Max(leftShoulder.y, rightShoulder.y));
        Vector2 bottomRight = new Vector2(Mathf.Max(leftHip.x, rightHip.x), Mathf.Min(leftHip.y, rightHip.y));

        Debug.Log($"Torso Bounding Box: TopLeft({topLeft.x},{topLeft.y}) BottomRight({bottomRight.x},{bottomRight.y})");

        GameObject line = new GameObject("TorsoBox");
        LineRenderer lr = line.AddComponent<LineRenderer>();
        lr.startWidth = 0.02f;
        lr.endWidth = 0.02f;
        lr.positionCount = 5;
        lr.useWorldSpace = true; // Fix: Use world space for proper positioning

        lr.SetPositions(new Vector3[]
        {
            new Vector3(topLeft.x, topLeft.y, 0),
            new Vector3(bottomRight.x, topLeft.y, 0),
            new Vector3(bottomRight.x, bottomRight.y, 0),
            new Vector3(topLeft.x, bottomRight.y, 0),
            new Vector3(topLeft.x, topLeft.y, 0)
        });
    }
}
