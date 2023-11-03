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
As the main goal of the project is an image classification task, we have chosen three objects that have distinct 3D structures and features. The 3D models are a [cardboard box](https://sketchfab.com/3d-models/small-cardboard-box-closed-9f0345c78b7b4761b9cdec5393474bd1), a [plate](https://sketchfab.com/3d-models/lunch-plate-school-project-eef24ebe601c4e2f99da3108ddc3b09b), and a [vase](https://sketchfab.com/3d-models/ancient-vase-dce37778ec964299bba5aeca736bf70e) and they were downloaded from [Sketchfab](https://sketchfab.com/).

![Untitled video - Made with Clipchamp](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/2205306a-c5cb-4178-897f-35f0625456b4)
![Untitled video - Made with Clipchamp (1)](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/c730440f-ce4b-4079-9ea7-cb9b2b2bfc2f)
![Untitled video - Made with Clipchamp (2)](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/0a39f30b-056d-4589-9fd3-b4741f6d15b5)

After downloading from Sketchfab, we import them on **Unity (2022.3.9.f1 version)**. We create a scene with a **camera**, a **light**, and a **plane**. We place our 3D object on the plane at ```(0,0,0)``` position and place the camera above the 3D object. We ensure in the game scene the object is not too far or too close to the camera. We also tilt the camera slightly, around 70 degrees, so that we can have a clear 3D view of the object. Below is an example of the setup:

<img width="889" alt="image" src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/90051c79-13f2-4ffc-8fb3-4dc071bbcf3b">

Note that the game scene represents the output image. We choose a resolution of ```400x400```. What we have demonstrated is how we can generate a synthetic image from 3D models using Unity. However, it's worth noting that we have only created a single image with a fixed object size and color, a uniform plane color, a specific camera position, and a singular lighting configuration. To build a robust classification model, we must acquire more data. Unfortunately, manually altering these parameters for each instance is a time-consuming and labor-intensive task. We must automate this process.

In order to generate large-scale synthetic datasets, we will use the [Perception Package](https://docs.unity3d.com/Packages/com.unity.perception@1.0/manual/index.html). More specifically, we will use the **Randomization tool** that allows us integrate domain randomization principles into your simulation.

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
