using UnityEngine;
using UnityEngine.UI; // Required if you add animations with CanvasGroup
using Mediapipe.Unity.Sample.PoseLandmarkDetection;
using Mediapipe.Unity;
using Mediapipe.Unity.Sample;
using System.Collections;
using System.Collections.Generic;

public class UiController : MonoBehaviour
{
    public GameObject homeScreenCanvas;

    public GameObject mainTryOnCanvas;

    public PoseLandmarkerRunner mediaPipeSolutionObject;

    private Stack<RectTransform> navigationStack = new Stack<RectTransform>();
    public RectTransform initialPanel;
    private bool isAnimating = false;


    void Start()
    {
        homeScreenCanvas.SetActive(true);
        mainTryOnCanvas.SetActive(false);
        mediaPipeSolutionObject.enabled = false;
        navigationStack.Push(initialPanel);
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

    public void pushUI(RectTransform panel)
    {
        if (isAnimating || panel == null)
        {
            return;
        }
        RectTransform panelToHide = null;
        if (navigationStack.Count > 0)
        {
            panelToHide = navigationStack.Peek();
        }
        navigationStack.Push(panel);

        StartCoroutine(SlideIn(panel, panelToHide));
    }
    public void popUI()
    {
        if (isAnimating || navigationStack.Count <= 1)
        {
            return;
        }
        RectTransform panelToPop = navigationStack.Pop();
        float parentWidth = ((RectTransform)panelToPop.parent).rect.width;
        panelToPop.offsetMin = new Vector2(-parentWidth, panelToPop.offsetMin.y);
        panelToPop.offsetMax = new Vector2(-parentWidth, panelToPop.offsetMax.y);
        panelToPop.gameObject.SetActive(false);
        RectTransform newTopPanel = navigationStack.Peek();
        newTopPanel.gameObject.SetActive(true);
    }
    private float duration = 0.4f;
    private IEnumerator SlideIn(RectTransform panelToShow, RectTransform panelToHide)
    {

        if (panelToShow.offsetMin.x == 0 && panelToShow.offsetMax.x == 0)
        {
            yield break;
        }
        isAnimating = true;

        float parentWidth = ((RectTransform)panelToShow.parent).rect.width;
        Vector2 startMin = new Vector2(-parentWidth, panelToShow.offsetMin.y);
        Vector2 startMax = new Vector2(-parentWidth, panelToShow.offsetMax.y);
        Vector2 endMin = new Vector2(0, panelToShow.offsetMin.y);
        Vector2 endMax = new Vector2(0, panelToShow.offsetMax.y);


        panelToShow.gameObject.SetActive(true);
        panelToShow.offsetMin = startMin;
        panelToShow.offsetMax = startMax;

        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.unscaledDeltaTime;
            float t = Mathf.Clamp01(elapsed / duration);
            float easedT = 1 - Mathf.Pow(1 - t, 3);

            panelToShow.offsetMin = Vector2.LerpUnclamped(startMin, endMin, easedT);
            panelToShow.offsetMax = Vector2.LerpUnclamped(startMax, endMax, easedT);

            yield return null;
        }

        panelToShow.offsetMin = endMin;
        panelToShow.offsetMax = endMax;

        if (panelToHide != null)
        {
            panelToHide.gameObject.SetActive(false);
        }
        isAnimating = false;
    }
}