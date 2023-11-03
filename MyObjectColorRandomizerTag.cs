using System;
using UnityEngine;
using UnityEngine.Perception.Randomization.Parameters;
using UnityEngine.Perception.Randomization.Randomizers;
using UnityEngine.Perception.Randomization.Samplers;

// This component should be added to any GameObject that you want to be randomized.
// The class name must match the name of the .cs file it is defined in.
public class MyObjectColorRandomizerTag : RandomizerTag { }

[Serializable]
// This attribute adds a menu entry in the Unity Editor for this randomizer.
[AddRandomizerMenu("MyObjectColorRandomizer")]
public class MyObjectColorRandomizer : Randomizer
{
    // Parameter for controlling the RGB color.
    public ColorRgbParameter colorRGB;
    
    // Reference to the material that will have its color randomized.
    public Material selectedMat;

    // This method is called at the start of each randomization iteration.
    protected override void OnIterationStart()
    {
        // Sample a random RGB color value.
        Color randomizedColor = colorRGB.Sample();

        // Apply the randomized color to the selected material.
        selectedMat.color = randomizedColor;
    }
}
