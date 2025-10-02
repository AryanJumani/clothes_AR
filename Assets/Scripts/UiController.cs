using UnityEngine;
using UnityEngine.UI; // Required if you add animations with CanvasGroup
using Mediapipe.Unity.Sample.PoseLandmarkDetection;
using Mediapipe.Unity;
using Mediapipe.Unity.Sample;
using System.Collections;
using System.Collections.Generic;
using UnityEngine.Networking;
using TMPro;

public class UiController : MonoBehaviour
{
    public GameObject homeScreenCanvas;

    public GameObject mainTryOnCanvas;

    public PoseLandmarkerRunner mediaPipeSolutionObject;

    private Stack<RectTransform> navigationStack = new Stack<RectTransform>();
    public RectTransform initialPanel;
    private bool isAnimating = false;

    [Header("Login/Register Input fields")]
    public TMP_InputField loginUsername;
    public TMP_InputField loginPassword;
    public TMP_InputField regUsername;
    public TMP_InputField regPassword;
    public TMP_InputField regConfirm;
    public RectTransform loginPanel;
    public RectTransform registerPanel;

    private void stackclear(RectTransform exception)
    {
        bool done = false;
        for (int i = 0; i < navigationStack.Count; i++)
        {
            RectTransform x = navigationStack.Pop();
            if (done == false && !x.Equals(exception))
            {
                done = true;
                x.gameObject.SetActive(false);
            }
        }
    }
    void Awake()
    {
        string jwtToken = PlayerPrefs.GetString("jwtToken", "");
        if (!string.IsNullOrEmpty(jwtToken))
        {
            Debug.Log("Auto-login with JWT token: " + jwtToken);
            homeScreenCanvas.SetActive(true);
            mainTryOnCanvas.SetActive(false);
            mediaPipeSolutionObject.enabled = false;
            registerPanel.gameObject.SetActive(false);
            navigationStack.Push(initialPanel);
            initialPanel.offsetMin = Vector2.zero;
            initialPanel.offsetMax = Vector2.zero;
        }
        else
        {
            Debug.Log("Register panel showing. No token found");
            homeScreenCanvas.SetActive(true);
            mainTryOnCanvas.SetActive(false);
            mediaPipeSolutionObject.enabled = false;
            navigationStack.Push(registerPanel);
        }
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
        StartCoroutine(animatePop(panelToPop));
        RectTransform newTopPanel = navigationStack.Peek();
        newTopPanel.gameObject.SetActive(true);
    }
    private float duration = 0.4f;
    private IEnumerator animatePop(RectTransform popPanel)
    {
        Vector3 startScale = popPanel.localScale;
        Vector3 endScale = new Vector3(2, 2, 2);
        TMP_Text[] children = popPanel.transform.GetComponentsInChildren<TMP_Text>();
        Image pop = popPanel.GetComponent<Image>();
        Color startColor = pop.color;
        Color endpopC = new Color(startColor.r, startColor.g, startColor.b, 0);
        Color endColor = new Color(1, 1, 1, 0);

        isAnimating = true;
        float elapsed = 0f;
        while (elapsed < duration)
        {
            elapsed += Time.unscaledDeltaTime;
            float t = Mathf.Clamp01(elapsed / duration);
            float easedT = 1 - Mathf.Pow(1 - t, 3);

            popPanel.localScale = Vector3.LerpUnclamped(startScale, endScale, easedT);
            pop.color = Color.LerpUnclamped(startColor, endpopC, easedT);
            foreach (TMP_Text child in children)
            {
                child.color = Color.LerpUnclamped(Color.white, endColor, easedT);
            }
            yield return null;
        }
        popPanel.gameObject.SetActive(false);
        popPanel.localScale = startScale;
        foreach (TMP_Text child in children)
        {
            child.color = Color.white;
        }
        pop.color = startColor;
        float parentWidth = ((RectTransform)popPanel.parent).rect.width;
        popPanel.offsetMin = new Vector2(-parentWidth, popPanel.offsetMin.y);
        popPanel.offsetMax = new Vector2(-parentWidth, popPanel.offsetMax.y);
        isAnimating = false;
    }
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
    public void Login()
    {
        StartCoroutine(LoginCor("http://127.0.0.1:5000/login"));
    }
    public void Register()
    {
        StartCoroutine(RegisterCor("http://127.0.0.1:5000/register"));
    }
    private IEnumerator LoginCor(string url)
    {
        User user = new User();
        user.username = loginUsername.text;
        user.password = loginPassword.text;
        string json = JsonUtility.ToJson(user);
        using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
        {
            byte[] raw = System.Text.Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(raw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            yield return request.SendWebRequest();
            if (request.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("Success...");
                string jsonResult = request.downloadHandler.text;
                string accesstoken = "";
                try
                {
                    var resultData = JsonUtility.FromJson<AccessTokenResponse>(jsonResult);
                    accesstoken = resultData.access_token;
                    PlayerPrefs.SetString("jwtToken", accesstoken);
                    PlayerPrefs.Save();
                }
                catch (System.Exception e)
                {
                    Debug.LogError("Could not parse access token from server response: " + e.Message);
                    ShowSnackbar("Error parsing server response", homeScreenCanvas.transform);
                    yield break;
                }
                ShowSnackbar("Login Success", homeScreenCanvas.transform);
                loginUsername.text = "";
                loginPassword.text = "";
                pushUI(initialPanel);
            }
            else
            {
                Debug.LogError("Error logging in");
                ShowSnackbar("Error", homeScreenCanvas.transform);
            }
        }
    }
    private IEnumerator RegisterCor(string url)
    {
        if (!regPassword.text.Equals(regConfirm.text))
        {
            Debug.Log("Passwords dont match");
            ShowSnackbar("Passwords dont match", homeScreenCanvas.transform);
            yield break;
        }
        User user = new User();
        user.username = regUsername.text;
        user.password = regPassword.text;
        string json = JsonUtility.ToJson(user);
        using (UnityWebRequest request = new UnityWebRequest(url, "POST"))
        {
            byte[] raw = System.Text.Encoding.UTF8.GetBytes(json);
            request.uploadHandler = new UploadHandlerRaw(raw);
            request.downloadHandler = new DownloadHandlerBuffer();
            request.SetRequestHeader("Content-Type", "application/json");
            yield return request.SendWebRequest();
            if (request.result == UnityWebRequest.Result.Success)
            {
                Debug.Log("Success...");
                ShowSnackbar("Registration Success", homeScreenCanvas.transform);
                regUsername.text = "";
                regPassword.text = "";
                regConfirm.text = "";
                pushUI(loginPanel);
            }
            else
            {
                Debug.LogError("Error registering user");
                ShowSnackbar("Error registering user", homeScreenCanvas.transform);
            }
        }
    }
    public void ShowSnackbar(string text, Transform parent)
    {
        GameObject snackbarObject = new GameObject("Snackbar (Temporary)");
        snackbarObject.transform.SetParent(parent.transform, false);

        Image panelImage = snackbarObject.AddComponent<Image>();
        panelImage.color = new Color(30f / 225f, 30f / 225f, 30f / 225f);

        RectTransform panelRect = snackbarObject.GetComponent<RectTransform>();
        panelRect.sizeDelta = new Vector2(parent.GetComponent<RectTransform>().rect.width - (20f * 2), 100f);
        panelRect.anchoredPosition = new Vector2(0, 75f);
        panelRect.anchorMin = new Vector2(0.5f, 0);
        panelRect.anchorMax = new Vector2(0.5f, 0);
        panelRect.pivot = new Vector2(0.5f, 0);
        GameObject textObject = new GameObject("SnackbarText");
        textObject.transform.SetParent(snackbarObject.transform, false);

        TextMeshProUGUI snackbarText = textObject.AddComponent<TextMeshProUGUI>();
        snackbarText.text = text;
        snackbarText.color = Color.white;
        snackbarText.fontSize = 30f;
        snackbarText.alignment = TextAlignmentOptions.Center;

        RectTransform textRect = textObject.GetComponent<RectTransform>();
        textRect.anchorMin = Vector2.zero; // Stretch to fill parent
        textRect.anchorMax = Vector2.one;
        textRect.sizeDelta = new Vector2(-20, 0); // Add some horizontal padding
        textRect.anchoredPosition = Vector2.zero;
        StartCoroutine(stopSnack(snackbarObject));
    }
    private IEnumerator stopSnack(GameObject objectToDestroy)
    {
        yield return new WaitForSeconds(1);
        Destroy(objectToDestroy);
    }

    public void Logout()
    {
        PlayerPrefs.DeleteKey("jwtToken");
        PlayerPrefs.Save();
        loginPanel.gameObject.SetActive(true);
        pushUI(loginPanel);
        stackclear(loginPanel);
        ShowSnackbar("Logout Successful, please log back in", mainTryOnCanvas.transform);
    }
}