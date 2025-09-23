using UnityEngine;
using UnityEngine.UI; // Required if you add animations with CanvasGroup

public class UiController : MonoBehaviour
{
    [Header("Canvases / UI Panels")]
    [Tooltip("The GameObject containing your home screen UI.")]
    public GameObject homeScreenCanvas;

    [Tooltip("The GameObject containing your main try-on UI (the bottom bar, etc.).")]
    public GameObject mainTryOnCanvas;

    [Header("Core App Components")]
    [Tooltip("The main GameObject that runs the MediaPipe Solution and starts the camera.")]
    public GameObject mediaPipeSolutionObject;

    void Start()
    {
        // 1. Set the initial state when the app launches.
        // Show the home screen.
        homeScreenCanvas.SetActive(true);

        // Make sure the main UI and the camera are turned OFF.
        mainTryOnCanvas.SetActive(false);
        mediaPipeSolutionObject.SetActive(false);
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

        // 2. Transition from the home screen to the main app.
        // Hide the home screen.
        homeScreenCanvas.SetActive(false);

        // Activate the main UI and the MediaPipe camera/tracking system.
        mainTryOnCanvas.SetActive(true);
        mediaPipeSolutionObject.SetActive(true);
    }
}