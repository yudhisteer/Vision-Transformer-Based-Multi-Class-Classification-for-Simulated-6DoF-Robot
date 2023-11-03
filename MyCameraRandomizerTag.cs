using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

// Add this Component to any GameObject that you would like to be randomized. This class must have an identical name to
// the .cs file it is defined in.
public class MyCameraRandomizerTag : RandomizerTag { }

[Serializable]
// This attribute adds a menu entry in the Unity Editor for this randomizer.
[AddRandomizerMenu("MyCameraRandomizer")]
public class MyCameraRandomizer : Randomizer
{
    // Parameters controlling the camera randomization.
    // The values of these parameters will be sampled during randomization.
    public FloatParameter cameraRotateX = new() { value = new UniformSampler(0, 1) };
    public FloatParameter cameraDistance = new() { value = new UniformSampler(0, 1) };

    // Reference to the camera that will be manipulated.
    public Camera myCamera;

    // Run this every randomization iteration
    protected override void OnIterationStart()
    {
        // Sample a random elevation and distance.
        float elevation = cameraRotateX.Sample();
        float distance = cameraDistance.Sample();

        // Calculate the camera's new position in a spherical coordinate system.
        float z = -distance * Mathf.Cos(elevation * Mathf.PI / 180);
        float y = distance * Mathf.Sin(elevation * Mathf.PI / 180);

        // Update the camera's rotation and position based on the sampled values.
        myCamera.transform.rotation = Quaternion.Euler(elevation, 0, 0);
        myCamera.transform.position = new Vector3(0, y, z);
    }
}
