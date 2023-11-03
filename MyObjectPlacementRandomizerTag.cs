using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

// This component should be added to any GameObject that you want to be randomized.
// The class name must match the name of the .cs file it is defined in.
public class MyObjectPlacementRandomizerTag : RandomizerTag { }

[Serializable]
// This attribute adds a menu entry in the Unity Editor for this randomizer.
[AddRandomizerMenu("MyObjectPlacementRandomizer")]
public class MyObjectPlacementRandomizer : Randomizer
{
    // Parameters for controlling object placement and properties.
    // These values will be sampled during randomization.
    public FloatParameter objScale;
    public Vector3Parameter placementRotation;
    public Vector3Parameter placementLocation;
    public IntegerParameter idObject;
    
    // Array of prefabs from which objects will be instantiated.
    public GameObject[] prefabs;

    // Reference to the current instance of the object being placed.
    public GameObject currentInstance;

    // This method is called at the start of each randomization iteration.
    protected override void OnIterationStart()
    {
        // Instantiate an object from the array of prefabs based on the sampled idObject.
        currentInstance = GameObject.Instantiate(prefabs[idObject.Sample()]);

        // Set the position, rotation, and scale of the instantiated object based on sampled values.
        currentInstance.transform.position = placementLocation.Sample();
        currentInstance.transform.eulerAngles = placementRotation.Sample();
        currentInstance.transform.localScale = Vector3.one * objScale.Sample();
    }

    // This method is called at the end of each randomization iteration to clean up the current object instance.
    protected override void OnIterationEnd()
    {
        GameObject.Destroy(currentInstance);
    }
}
