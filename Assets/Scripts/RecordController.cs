using UnityEngine;
using UnityEngine.Events;

public class RecordController : MonoBehaviour
{
    private bool isRecording = false;

    public static UnityAction onAllowCallback;
    public static UnityAction onDenyCallback;
    public static UnityAction onDenyAndNeverAskAgainCallback;

    private AndroidJavaObject androidRecorder;

    void Start()
    {
        DontDestroyOnLoad(gameObject);

        using (AndroidJavaClass unityClass = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
        {
            androidRecorder = unityClass.GetStatic<AndroidJavaObject>("currentActivity");

            androidRecorder.Call("setUpSaveFolder", "MyGameRecordings");

            int width = Screen.width;
            int height = Screen.height;
            int bitrate = 3000 * 1000;
            int fps = 30;
            bool audioEnable = false;

            androidRecorder.Call("setupVideo", width, height, bitrate, fps, audioEnable, "H264");
        }
    }
    public void ToggleRecording()
    {
        if (isRecording)
        {
            StopVideoRecording();
        }
        else
        {
            StartVideoRecording();
        }
    }

    private void StartVideoRecording()
    {
        Debug.Log("Starting recording...");
        androidRecorder.Call("startRecording");
    }

    private void StopVideoRecording()
    {
        Debug.Log("Stopping recording...");
        androidRecorder.Call("stopRecording");
    }
    public void VideoRecorderCallback(string message)
    {
        Debug.Log("VideoRecorderCallback received message: " + message);
        switch (message)
        {
            case "start_record":
                isRecording = true;
                // TODO: change icon to stop icon
                break;
            case "stop_record":
                isRecording = false;
                // TODO: change back to normal
                break;
            case "init_record_error":
                Debug.LogError("Recorder failed to initialize!");
                isRecording = false;
                break;
        }
    }
}