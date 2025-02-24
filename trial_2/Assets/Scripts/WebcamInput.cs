using UnityEngine;
using UnityEngine.UI;
public class WebcamInput : MonoBehaviour
{
    public RawImage webcamDisplay;
    private WebCamTexture webcamTexture;

    void Start()
    {
        WebCamDevice[] devices = WebCamTexture.devices;
        if (devices.Length > 0)
        {
            webcamTexture = new WebCamTexture(devices[0].name);
            webcamTexture.Play();
            webcamDisplay.texture = webcamTexture;
        }
        else
        {
            Debug.LogError("nothing detected");
        }
    }

    public Texture2D GetFrame()
    {
        if (webcamTexture == null || !webcamTexture.isPlaying) return null;

        Texture2D frame = new Texture2D(webcamTexture.width, webcamTexture.height);
        frame.SetPixels(webcamTexture.GetPixels());
        frame.Apply();
        return frame;
    }
}
