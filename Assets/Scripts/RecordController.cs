using UnityEngine;
using System.IO;
using System;
public enum VideoEncoder
{
    DEFAULT,
    H263,
    H264,
    HEVC,
    MPEG_4_SP,
    VP8
}


public class RecordController : MonoBehaviour
{
    public GameObject recordBtn;
    public GameObject stopBtn;
    private bool isRecording = false;

    private AndroidJavaObject androidRecorder;

    void Start()
    {
        DontDestroyOnLoad(gameObject);

#if UNITY_ANDROID && !UNITY_EDITOR
        using (AndroidJavaClass unityClass = new AndroidJavaClass("com.unity3d.player.UnityPlayer"))
        {
            // Get the current Activity, which is our AndroidUtils class
            androidRecorder = unityClass.GetStatic<AndroidJavaObject>("currentActivity");

            // Call the setup methods in your AndroidUtils.java
            androidRecorder.Call("setUpSaveFolder", "MyRecordings");

            int width = Screen.width;
            int height = Screen.height;
            int bitrate = 6*1000*1000;
            int fps = 30;
            bool audioEnable = true;

            // androidRecorder.Call("setupVideo", width, height, bitrate, fps, audioEnable, VideoEncoder.H264.ToString()); // commenting to see if default records best.
        }
#endif
        recordBtn.SetActive(true);
        stopBtn.SetActive(false);
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
        Debug.Log("Sending start recording command...");
#if UNITY_ANDROID && !UNITY_EDITOR
        androidRecorder.Call("startRecording");
#endif
    }

    private void StopVideoRecording()
    {
        Debug.Log("Sending stop recording command...");
#if UNITY_ANDROID && !UNITY_EDITOR
        androidRecorder.Call("stopRecording");
#endif
    }

    public void VideoRecorderCallback(string message)
    {
        Debug.Log("VideoRecorderCallback received message: " + message);
        if (string.IsNullOrEmpty(message)) return;


        if (message.StartsWith("start_record"))
        {
            isRecording = true;
            recordBtn.SetActive(false);
            stopBtn.SetActive(true);
            Debug.Log("State updated: Recording has started.");
        }
        else if (message.StartsWith("stop_record"))
        {
            isRecording = false;
            recordBtn.SetActive(true);
            stopBtn.SetActive(false);
        }
        else if (message == "init_record_error")
        {
            Debug.LogError("Recorder failed to initialize!");
            isRecording = false;

            stopBtn.SetActive(false);
            recordBtn.SetActive(true);
        }
    }
}