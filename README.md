# Vision Transformer-Based Multi-Class Classification in Simulated 6DoF Robot Environment

## Problem Statement

## Dataset

## Abstract

## Plan of Action
1. [Generating Synthetic Dataset with Unity](#syntehtic)
2. [A Gentle Introduction to Attention](#attention)
3. [Vision Transformer: How much is an image worth?](#vision)
4. [Coding Transformers for Image Recognition: From Pixels to Predictions](#transformer)
5. [Simulating Palletizing with Transformers](#simulation)

-----------------
<a name="syntehtic"></a>
## 1. Generating Synthetic Dataset with Unity

### 1.1 Object Classes

![Untitled video - Made with Clipchamp](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/2205306a-c5cb-4178-897f-35f0625456b4)
![Untitled video - Made with Clipchamp (1)](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/c730440f-ce4b-4079-9ea7-cb9b2b2bfc2f)
![Untitled video - Made with Clipchamp (2)](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/0a39f30b-056d-4589-9fd3-b4741f6d15b5)

### 1.2 Camera Randomizer

```c#
    // Sample a random elevation and distance.
    float elevation = cameraRotateX.Sample();
    float distance = cameraDistance.Sample();

    // Calculate the camera's new position in a spherical coordinate system.
    float z = -distance * Mathf.Cos(elevation * Mathf.PI / 180);
    float y = distance * Mathf.Sin(elevation * Mathf.PI / 180);

    // Update the camera's rotation and position based on the sampled values.
    myCamera.transform.rotation = Quaternion.Euler(elevation, 0, 0);
    myCamera.transform.position = new Vector3(0, y, z);
```

### 1.3 Plane Color Randomizer

```c#
    // Sample a random RGB color value.
    Color randomizedColor = colorRGB.Sample();

    // Apply the randomized color to the selected material.
    selectedMat.color = randomizedColor;
```

### 1.4 Light Randomizer

```c#
    // Randomize the rotation of the light using the sampled values.
    tagLight.transform.eulerAngles = new Vector3(lightRotateX.Sample(), lightRotateY.Sample(), lightRotateZ.Sample());
```

### 1.5 Object Placement Randomizer

```c#
    // Set the position, rotation, and scale of the instantiated object based on sampled values.
    currentInstance.transform.position = placementLocation.Sample();
    currentInstance.transform.eulerAngles = placementRotation.Sample();
    currentInstance.transform.localScale = Vector3.one * objScale.Sample();
```


### 1.6 Object Color Randomizer

```c#
    // Sample a random RGB color value.
    Color randomizedColor = colorRGB.Sample();

    // Apply the randomized color to the selected material.
    selectedMat.color = randomizedColor;
```



### 1.7 Simulation



<img width="1077" alt="image" src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/d929d63a-8e0a-4042-8330-9f61394c7eb8">
<img width="1073" alt="image" src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/2b72a160-35e9-435a-a67a-59dccd91e7da">
<img width="1073" alt="image" src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/7ce21473-5ed3-40bd-8407-6761ad2c82b2">


-----------------
<a name="attention"></a>
## 2. A Gentle Introduction to Attention

-----------------
<a name="vision"></a>
## 3. Vision Transformer: How much is an image worth?

-----------------
<a name="transformer"></a>
## 4. Coding Transformers for Image Recognition: From Pixels to Predictions

-----------------
<a name="simulation"></a>
## 5. Simulating Palletizing with Transformers









----------
## Conclusion

## References
1. https://arxiv.org/abs/2010.11929
2. https://www.youtube.com/watch?v=TrdevFK_am4&ab_channel=YannicKilcher
3. https://www.youtube.com/watch?v=j3VNqtJUoz0&ab_channel=DeepFindr
4. https://www.youtube.com/watch?v=DVoHvmww2lQ&list=PLpZBeKTZRGPMddKHcsJAOIghV8MwzwQV6&index=1&ab_channel=AICoffeeBreakwithLetitia
5. https://www.youtube.com/watch?v=j6kuz_NqkG0&ab_channel=AleksaGordi%C4%87-TheAIEpiphany
6. 
