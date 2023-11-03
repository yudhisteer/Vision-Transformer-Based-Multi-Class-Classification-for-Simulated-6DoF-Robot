using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

[RequireComponent(typeof(Light))]
// This component should be added to any GameObject that you want to be randomized.
// The class name must match the name of the .cs file it is defined in.
public class MyLightRandomizerTag : RandomizerTag { }

[Serializable]
// This attribute adds a menu entry in the Unity Editor for this randomizer.
[AddRandomizerMenu("MyLightRandomizer")]
public class MyLightRandomizer : Randomizer
{
    // Parameters controlling the randomization of the light.
    // These values will be sampled during randomization.
    public FloatParameter lightIntensity = new() { value = new UniformSampler(0, 1) };
    public FloatParameter lightRotateX = new() { value = new UniformSampler(0, 1) };
    public FloatParameter lightRotateY = new() { value = new UniformSampler(0, 1) };
    public FloatParameter lightRotateZ = new() { value = new UniformSampler(0, 1) };

    // This method is called at the start of each randomization iteration.
    protected override void OnIterationStart()
    {
        // Query for all GameObjects in the scene with the MyLightRandomizerTag component.
        var tags = tagManager.Query<MyLightRandomizerTag>();
        foreach (var tag in tags)
        {
            // Get the Light component attached to the object with the MyLightRandomizerTag.
            var tagLight = tag.GetComponent<Light>();
            
            // Randomize the rotation of the light using the sampled values.
            tagLight.transform.eulerAngles = new Vector3(lightRotateX.Sample(), lightRotateY.Sample(), lightRotateZ.Sample());
        }
    }
}
