using UnityEngine;
using UnityEngine.UI; // Required if you add animations with CanvasGroup
using Mediapipe.Unity.Sample.PoseLandmarkDetection;
using Mediapipe.Unity;
using Mediapipe.Unity.Sample;
using System.Collections;

public class UiController : MonoBehaviour
{
    public GameObject homeScreenCanvas;

    public GameObject mainTryOnCanvas;

    public PoseLandmarkerRunner mediaPipeSolutionObject;

    void Start()
    {
        homeScreenCanvas.SetActive(true);
        mainTryOnCanvas.SetActive(false);
        mediaPipeSolutionObject.enabled = false;
    }

    /// <summary>
    /// This public function will be called by your "START LIVE TRY-ON" button.
    /// </summary>
    public void StartTryOnExperience()
    {
        // Check if all objects are assigned to prevent errors.
        if (homeScreenCanvas == null || mainTryOnCanvas == null || mediaPipeSolutionObject == null)
        {
            Debug.LogError("Not all GameObjects are assigned in the AppStartController Inspector!");
            return;
        }

        mainTryOnCanvas.SetActive(true);
        homeScreenCanvas.SetActive(false);


        mediaPipeSolutionObject.enabled = true;
        StartCoroutine(SelectFront());
    }

    private IEnumerator SelectFront()
    {
        ImageSource imageSource = ImageSourceProvider.ImageSource;
        while (imageSource == null)
        {
            Debug.LogWarning("Waiting for ImageSource to be ready...");
            yield return null; // Wait for the next frame
            imageSource = ImageSourceProvider.ImageSource; // Try to get it again
        }

        var cameraDevices = imageSource.sourceCandidateNames;

        if (cameraDevices == null || cameraDevices.Length == 0)
        {
            Debug.LogError("No camera devices found!");
            yield break;
        }
        WebCamDevice[] devices = WebCamTexture.devices;

        if (devices.Length == 0)
        {
            Debug.LogError("No camera devices found!");
            yield break;
        }
        for (int i = 0; i < devices.Length; i++)
        {
            if (devices[i].isFrontFacing)
            {
                Debug.Log($"Front-facing camera found: {devices[i].name} at index {i}");
                imageSource.SelectSource(i);
            }
        }

        Debug.LogWarning("Could not find a front-facing camera. Using default camera.");
    }
}