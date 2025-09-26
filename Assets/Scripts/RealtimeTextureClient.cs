using UnityEngine;
using UnityEngine.Networking; // Required for UnityWebRequest
using System.Collections;
using System.IO;
using System;
#if UNITY_STANDALONE || UNITY_EDITOR
// using SFB;
#endif
#if !UNITY_EDITOR && UNITY_ANDROID
using UnityEngine.Android;
#endif


public class RealtimeTextureController : MonoBehaviour
{
    [Header("Component References")]
    public ARAttach arAttachScript;
    public UiController uiController;

    [Header("Server Configuration")]
    public string serverIpAddress = "127.0.0.1";

    private Material runtimeMaterial;
    private string serverUrl;

    [Serializable] private class ImageRequest { public string image; }
    [Serializable] private class ImageResponse { public string image; }

    void Awake()
    {
        // This block is a good practice for ensuring the plugin requests
        // the necessary permissions on modern Android versions.
#if UNITY_ANDROID && !UNITY_EDITOR
        using (AndroidJavaClass ajc = new AndroidJavaClass("com.yasirkula.unity.NativeGalleryMediaPickerFragment"))
        {
            ajc.SetStatic<bool>("GrantPersistableUriPermission", true);
        }
#endif

#if UNITY_ANDROID && !UNITY_EDITOR
        serverUrl = $"http://{serverIpAddress}:5000/process_image";
#else
        serverUrl = "http://127.0.0.1:5000/process_image";
#endif
    }

    public void OnUploadButtonClick()
    {
#if UNITY_ANDROID && ! UNITY_EDITOR
        RequestPermissionAndPickImage();
// #elif UNITY_STANDALONE || UNITY_EDITOR
//         var extensions = new[] { new ExtensionFilter("Image Files", "png", "jpg") };
//         StandaloneFileBrowser.OpenFilePanelAsync("Select Design", "", extensions, false, (string[] paths) => {
//             if (paths.Length > 0 && !string.IsNullOrEmpty(paths[0]))
//             {
//                 byte[] fileBytes = File.ReadAllBytes(paths[0]);
//                 StartCoroutine(SendImageToServer(fileBytes));
//             }
//         });
#elif UNITY_EDITOR
        string imagePath = "/Users/ajgamergrenade/projects_and_trials/clothes_AR/trial2.png";
        Debug.Log($"EDITOR MODE: Attempting to read file from hardcoded path: {imagePath}");
        try
        {
            byte[] fileBytes = File.ReadAllBytes(imagePath);
            StartCoroutine(SendImageToServer(fileBytes));
        }
        catch (Exception e)
        {
            Debug.LogError($"FAILED TO READ FILE. Check the path is correct. Error: {e.Message}");
        }
#endif
    }

    private async void RequestPermissionAndPickImage()
    {
        NativeGallery.Permission permission = await NativeGallery.RequestPermissionAsync(NativeGallery.PermissionType.Read, NativeGallery.MediaType.Image);

        if (permission == NativeGallery.Permission.Granted)
        {
            NativeGallery.GetImageFromGallery((path) =>
            {
                if (!string.IsNullOrEmpty(path))
                {
                    // The path is a "content://" URI.
                    // We will use a UnityWebRequest to load it, which correctly handles these URIs.
                    StartCoroutine(LoadImageFromUri(path));
                }
                else
                {
                    Debug.Log("User cancelled image selection.");
                }
            }, "Select Design", "image/*");
        }
        else
        {
            Debug.LogError("User denied gallery access.");
        }
    }

    private IEnumerator LoadImageFromUri(string uri)
    {
        uri = "file://" + uri;
        Debug.Log("Attempting to load image from URI: " + uri);
        using (UnityWebRequest www = UnityWebRequestTexture.GetTexture(uri))
        {
            yield return www.SendWebRequest();

            if (www.result == UnityWebRequest.Result.Success)
            {
                Texture2D texture = DownloadHandlerTexture.GetContent(www);
                if (texture != null)
                {
                    Debug.Log("Successfully loaded texture from URI.");
                    // Continue the process by sending the image to the server
                    byte[] imageBytes = texture.EncodeToPNG();
                    StartCoroutine(SendImageToServer(imageBytes));
                }
                else
                {
                    Debug.LogError("Failed to get texture content from web request.");
                }
            }
            else
            {
                Debug.LogError($"Failed to load image from URI: {uri}. Error: {www.error}");
            }
        }
    }

    private IEnumerator SendImageToServer(byte[] imageBytes)
    {
        ImageRequest payload = new ImageRequest { image = Convert.ToBase64String(imageBytes) };
        string jsonPayload = JsonUtility.ToJson(payload);
        byte[] bodyRaw = System.Text.Encoding.UTF8.GetBytes(jsonPayload);

        UnityWebRequest request = new UnityWebRequest(serverUrl, "POST");
        request.uploadHandler = new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");

        Debug.Log("Sending image to Python server...");
        yield return request.SendWebRequest();

        if (request.result == UnityWebRequest.Result.Success)
        {
            uiController.StartTryOnExperience();
            ImageResponse response = JsonUtility.FromJson<ImageResponse>(request.downloadHandler.text);
            byte[] processedBytes = Convert.FromBase64String(response.image);
            StartCoroutine(ApplyTextureWhenReady(processedBytes));
        }
        else
        {
            Debug.LogError($"Error from server ({serverUrl}): {request.error}");
        }
    }

    private IEnumerator ApplyTextureWhenReady(byte[] textureData)
    {
        // Wait until the AR object has been instantiated
        while (arAttachScript.TshirtInstance == null)
        {
            yield return null;
        }

        // Create a new material instance if one doesn't exist
        if (runtimeMaterial == null)
        {
            Renderer tshirtRenderer = arAttachScript.TshirtInstance.GetComponentInChildren<Renderer>();
            runtimeMaterial = new Material(tshirtRenderer.material);
            tshirtRenderer.material = runtimeMaterial;
        }

        Texture2D tex = new Texture2D(2, 2);
        tex.LoadImage(textureData);
        runtimeMaterial.mainTexture = tex;
        Debug.Log("New real-time texture applied!");
    }
}

