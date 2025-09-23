using UnityEngine;
using UnityEngine.Android;
using System.Collections;
using System.Collections.Generic;

public class RequestAllPermissions : MonoBehaviour
{
    void Start()
    {
        StartCoroutine(RequestAllListedPermissions());
    }

    private IEnumerator RequestAllListedPermissions()
    {
        // A list of common permissions an app might request.
        // WARNING: Only include permissions your app actually needs for production.
        List<string> permissionsToRequest = new List<string>
        {
            Permission.Camera,
            Permission.ExternalStorageRead,
            Permission.ExternalStorageWrite
        };

        foreach (string permission in permissionsToRequest)
        {
            if (!Permission.HasUserAuthorizedPermission(permission))
            {
                Debug.Log($"Requesting permission for: {permission}");
                Permission.RequestUserPermission(permission);

                // The permission dialog is asynchronous. We'll wait a moment here.
                // A more robust implementation would use a callback system.
                yield return new WaitForSeconds(1);
            }
        }
    }
}