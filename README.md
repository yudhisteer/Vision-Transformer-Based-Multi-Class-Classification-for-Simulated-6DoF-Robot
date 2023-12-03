# Vision Transformer-Based Multi-Class Classification for Simulated 6DoF Robot

## Problem Statement

We want to deploy a 6-DOF robotic arm in an industry that will be able to pick up objects with a certain grip and place the object in their designated box. Deploying a 6-DOF robotic arm in an industry requires a lot of **planning** and faces multiple challenges. For example, we may want to plan the **trajectory** of the movement of the robot beforehand to ensure we have the **space** required for the machine but also to ensure the **safety** of the **machine** and the **people** around it. 

For our problem statement, we may want the robot to recognize the object first. To do that, we will need to collect **images** of the objects. But how many images do we need, and who's responsible for taking them? Can we rely on internet images, or do we risk them not truly representing our objects?

Furthermore, optimizing the robot's efficiency might involve **redesigning** the **workspace**. Is it necessary to physically place the robot in a real setting to test for maximum efficiency? And if so, do we have to halt other machines' work to make room for the robot? 

So how can we test our project in a **simulated** **setting** that can efficiently help us plan the **end-to-end tasks** the robot will need to perform? We will want to test our object classification in this simulation as well so that the images truly represent the real-world settings in which the robot operates.


## Dataset
- A synthetic dataset was created using the ```Unity Perception``` package. The dataset of ```150``` images for each class - **Box**, **Plate**, and **Vase** - can be found on [Kaggle](https://www.kaggle.com/datasets/yudhisteerchintaram/synthetic-image-dataset-box-plate-and-vase/data).

## Abstract

The project involves building an object classification AI model using Vision Transformer. Instead of taking pictures manually or scraping images off the internet, we generated synthetic images of the objects we wanted to classify. Using the Unity Perception package, we wrote a script that will position the object randomly on a plane of random color. We also position the camera inside Unity at a random position and orientation for each frame. We do the same for the lighting. Finally, we want the object to be of random color and random size so that we have a larger variability of data to build a model of minimum bias. We then built a Vision Transformer model from scratch and trained on 150 images for each object with a test size ratio of 20%. However, from the ViT paper, we concluded that training a Transformer model from scratch on a small dataset achieves inferior results. We use a pre-trained ViT model from PyTorch and perform a transfer learning approach to train on our custom synthetic dataset. We obtained a test and train accuracy of 100%. 

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/91659faa-6897-4dda-b48e-d06f3105f581" controls="controls" style="max-width: 730px;">
  </video>
</div>

We build a ```Digital Twin``` of the workspace we want to place our 6 DOF robot in Unity. We have three conveyor belts and at the end of each, we have a container that will be used to collect the objects (box, plate, or, vase). We place markers on the main conveyor belt to mark the position at which the object will stop in front of the robot. We create another camera object on top of the markers that will be used to capture images of the randomly instantiated object. This image will then go through our ViT model to classify the object. Based on the object, the robot will choose an appropriate gripper to grab the object and place it on its respective conveyor belt. This process will then repeat.

This process shows a cheaper, faster, and more effective way of testing end-to-end tasks with a 6DOF robot. With large synthetic datasets, we cut time and manpower in collecting data for our AI model however, we need to make sure these data are representative of the data in real-world. By using the AI model in a Digital Twin, we get an idea what are the challenges we may face when implementing the robot in the industry and we can run as many tests as we want without hampering other processes.



## Plan of Action
1. [Generating Synthetic Dataset with Unity](#synthetic)
2. [Vision Transformer: How much is an image worth?](#vision)
3. [Coding Transformers for Image Recognition: From Pixels to Predictions](#transformer)
4. [Digital Twin with Transformers](#simulation)

-----------------
<a name="synthetic"></a>
## 1. Generating Synthetic Dataset with Unity

### 1.1 Object Classes
As the main goal of the project is an image classification task, we have chosen three objects that have distinct 3D structures and features. The 3D models are a [cardboard box](https://sketchfab.com/3d-models/small-cardboard-box-closed-9f0345c78b7b4761b9cdec5393474bd1), a [plate](https://sketchfab.com/3d-models/lunch-plate-school-project-eef24ebe601c4e2f99da3108ddc3b09b), and a [vase](https://sketchfab.com/3d-models/ancient-vase-dce37778ec964299bba5aeca736bf70e) and they were downloaded from [Sketchfab](https://sketchfab.com/).

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/2205306a-c5cb-4178-897f-35f0625456b4" width="30%" />
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/c730440f-ce4b-4079-9ea7-cb9b2b2bfc2f" width="30%" />
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/0a39f30b-056d-4589-9fd3-b4741f6d15b5" width="30%" />
</p>

After downloading from Sketchfab, we import them on **Unity (2022.3.9.f1 version)**. We create a scene with a **camera**, a **light**, and a **plane**. We place our 3D object on the plane at ```(0,0,0)``` position and place the camera above the 3D object. We ensure in the game scene the object is not too far or too close to the camera. We also tilt the camera slightly, around 70 degrees, so that we can have a clear 3D view of the object. Below is an example of the setup:

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/90051c79-13f2-4ffc-8fb3-4dc071bbcf3b" width="90%" />
</p>


Note that the game scene represents the output image. We choose a resolution of ```400x400```. What we have demonstrated is how we can generate a synthetic image from 3D models using Unity. However, it's worth noting that we have only created a single image with a fixed object size and color, a uniform plane color, a specific camera position, and a singular lighting configuration. To build a robust classification model, we must acquire more data. Unfortunately, manually altering these parameters for each instance is a time-consuming and labor-intensive task. We must automate this process.

In order to generate large-scale synthetic datasets, we will use the [Perception Package](https://docs.unity3d.com/Packages/com.unity.perception@1.0/manual/index.html). More specifically, we will use the **Randomization tool** that allows us to integrate domain randomization principles into our simulation.

### 1.2 Camera Randomizer
We start by creating a script that will randomly position and rotate our camera in the scene. We will randomly select a value for the distance (about the y-axis) and the elevation (about the z-axis). We then calculate the camera's new position in a spherical coordinate system using the elevation and distance values. We update the camera's rotation and position based on the sampled values. We set the camera distance to be between ```5``` and ```8``` units whereas the rotation about the x-axis to be between ```15``` and ```95``` degrees. 

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
Next, we want our plane to change color for each frame. We create a new script that will randomly sample an  RGB color value. We then apply that color to the plane. Note that our script will iterate for each frame depending on the number of iterations we set. the colors could be any values between ```0``` and ```255```.

```c#
    // Sample a random RGB color value.
    Color randomizedColor = colorRGB.Sample();

    // Apply the randomized color to the selected material.
    selectedMat.color = randomizedColor;
```

### 1.4 Light Randomizer
Similarly, we want the lighting in our sample to be different for each frame. We randomly sample rotation values in the x,y, and z coordinates and apply them to the lighting object. We set the light intensity to change between ```0.5``` and ```3```. The rotation about the x-axis should be between ```15``` and ```90``` degrees, about the y-axis should be between ```-180``` and ```180``` and about the z-axis should be between ```0``` and ```1``` degrees. 

```c#
    // Randomize the rotation of the light using the sampled values.
    tagLight.transform.eulerAngles = new Vector3(lightRotateX.Sample(), lightRotateY.Sample(), lightRotateZ.Sample());
```

### 1.5 Object Placement Randomizer
Furthermore, we do not want our object to be fixed in the center of the plane. We want the object to have a new **position**, **rotation**, and **scale** for each frame. We thus sample values for these three properties and apply them to the instantiated object. We set the scale of the object to vary between ```0.8``` and ```1.2```, the placement rotation to be between ```-180``` and ```180``` about the y-axis only, and the placement location to vary between ```-1.8``` and ```+1.8```.

```c#
    // Set the position, rotation, and scale of the instantiated object based on sampled values.
    currentInstance.transform.position = placementLocation.Sample();
    currentInstance.transform.eulerAngles = placementRotation.Sample();
    currentInstance.transform.localScale = Vector3.one * objScale.Sample();
```


### 1.6 Object Color Randomizer
Lastly, we want our object to be of different colors for each iteration. Similarly, to the plane color randomizer, we sample random RGB values and apply them to our object. We need to make sure we are using the color material associated with our prefab 3D object.

```c#
    // Sample a random RGB color value.
    Color randomizedColor = colorRGB.Sample();

    // Apply the randomized color to the selected material.
    selectedMat.color = randomizedColor;
```



### 1.7 Simulation
We then run the simulation for 150 iterations. Note that we create a new scene for each object. We tried to spawn the 3D objects randomly one at a time in one scene only however, all images will be stored in one folder and it defeats the purpose of building the image classification if we manually need to filter the images at this stage. It can be a good exercise to test our model with this random spawned 3D model approach in the end though.  Below is a step-by-step output of the result of the simulation:

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/0ada2f9b-20a7-4069-89c4-30bceff2b2aa" controls="controls" style="max-width: 730px;">
  </video>
</div>

We opted for an iteration count of ```150```. In the image classification format, images are organized into distinct directories labeled with specific class names. For instance, all images of plates are stored in the "plate/" directory. This format is widely used in various image classification benchmarks, such as ImageNet. Below, you can find an example of this storage format, with arbitrary image numbers.

```text
cardboard_plate_vase/ <- overall dataset folder
    train/ <- training images
        cardboard/ <- class name as folder name
            image01.jpeg
            image02.jpeg
            ...
        plate/
            image24.jpeg
            image25.jpeg
            ...
        vase/
            image37.jpeg
            ...
    test/ <- testing images
        cardboard/
            image101.jpeg
            image102.jpeg
            ...
        plate/
            image154.jpeg
            image155.jpeg
            ...
        vase/
            image167.jpeg
            ...
```

Below is the output of the images for the plate object class:

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/7ce21473-5ed3-40bd-8407-6761ad2c82b2" />
</p>

-----------------
<a name="vision"></a>
## 2. Vision Transformer: How much is an image worth?
Most of the explanation found below comes from the Vision Transformer paper itself: [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
](https://arxiv.org/abs/2010.11929)

### 2.1 Overview
Before we dive into Transformer, let's do a quick overview of how CNN works first. 

1. A Convolutional Neural Network (CNN) uses **kernels** to gather **local** information in each layer.
2. This information is subsequently passed to the next layer, which further aggregates local data with an enlarged **receptive field**.
3. The latter occurs because the final layer considers information that has already been **aggregated** by the initial layer.
4. Initially, CNNs focus on local areas to capture **simple patterns** such as **edges**, and as they progress through the layers, their receptive fields become more **global**. This enables them to capture more complex patterns and semantics.

The author of the ViT paper claims that they did not rely on attention in conjunction with CNN but a solely Transformer architecture was used directly to a sequence of patches of images that outperformed SOTA image classification models. Self-attention-based architectures - Transformers - have been the SOTA method for many NLP tasks. However, CNN remained dominant in computer vision. In order to apply, the Transformer network to a computer vision task, they had to transform their input data (images) the same way the input (tokens) was for a Transformer network in NLP. They split the image into patches of ```16x16``` pixels. These patches are then flattened by a linear transformation matrix to become vectors. The vector of each patch gets a positional embedding and a Transformer is used to predict the class of the image. 

However, the ViT model underperformed when trained on mid-size datasets because of a lack of "**inductive bias**" (I later explain this term). But when trained on **larger datasets** (```14M-300M```) images, they outperformed SOTA image recognition models and concluded that "_large  scale training trumps inductive bias_". 

In summary:

- The Transformer requires **abundant data** because it has the freedom to look **everywhere** at the image from the start. That is, it is unfocused at the start and needs a huge amount of data to learn **what** and **where** to focus to make the right predictions.
- The Transformer can find **novel** ways to look at the data because it isn't guided on how to do so.
- On the other hand, CNN is focused in the beginning by the convolutions towards a **local view**. The given focus patterns can be a limitation but we spare a lot of training data because we do not have to teach the model how to focus but only **where** to focus. 



### 2.2 The Architecture



<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/51e48470-3a5a-4d2f-8b4c-91c53f1a607a" />
</p>

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/bedb311c-2dea-47ec-a5d5-a0485e436574" width="60%"/>
</p>


### 2.3 Equation 1
The author explains that the original Transformer for the NLP task takes a sequence of token embeddings as input. Hence, we need to transform our images into the same. 

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/4c7c5917-41ff-4aef-b0eb-09d36834bc75" width="60%"/>
</p>


<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/3d13cde2-d641-4d27-a2e7-aaf3d80c5a27" />
</p>

where

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/71ffefd2-a534-41a6-b165-99a5c36c786a" />
</p>

and

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/86bd48b5-aca0-4844-8f34-7ecfc8052ec3" />
</p>


The representation of our input is to split an image into fixed-size patches and then return each patch of the image into a **learnable embedding**. Our original image is of size ```224 x 224 x 3 -> [Height x Width x Color Channel]```. We choose a patch size of ```16 x 16```, therefore the resulting number of patches is:

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/f5622088-ad7e-4fe3-8d7b-0fb9d208eed8" />
</p>

Hence, the output shape of a single 2D image flattened into patches will be of size: ```196 x 768 -> [N x (P^2 ⋅ C)]```. Note that ```196``` is the input sequence length for the transformer and ```768``` is the embedding dimension.


<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/39fa623a-d35a-4272-9735-0f0f4a8734f0" width="90%"/>
</p>

Next, we need to create a **learnable class token** with one value for each of the embedding dimensions and then prepend it to the original sequence of patch embeddings. Our class token will be of size ```1 x 768```. Note that one of the authors of the paper, [Lucas Beyer](https://github.com/google-research/vision_transformer/issues/61#issuecomment-802233921),  stated that the ```"'extra class embedding' is not really important. However, we wanted the model to be "exactly Transformer, but on image patches", so we kept this design from Transformer, where a token is always used."```

Our flattened patches with the prepend class encoding will be of size ```197 x 768```. Next, we need to add a learnable ```1D``` positional embedding to capture **positional information**. Now, why is this important? Since we are splitting our ```2D``` image into ```1D``` patches, we still want to retain the **order** of the patches. This will help us understand which patch relates to which patch. In order to add the positional embedding to our existing embedded patches with a prepend class token, the size of the positional embedding should also be ```197 x 768```.



### 2.4 Equation 2
Equation 2 involves coding the first two blocks in the Transformer Encoder: **Layer Normalization (LN)** and the **Multi-Head Self Attention (MSA)** layer. 

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/a0f48f93-f24d-428f-8afc-bfd26459499e" width="50%"/>
</p>

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/4cae8e1e-8c62-49e8-b17a-01e4728de0d3" width="25%"/>
</p>


- Linear Normalization (LN) helps stabilize the training process and improve generalization performance by normalizing the activations of each layer in the model independently. It uses the mean and standard deviation of the activations for each layer and then transforms them into a **normal distribution**. In summary, LN prevents **exploding** or **vanishing gradients**, which can make it difficult to train deep neural networks.

- Multi-Head Self Attention (MSA) provides a mechanism for patches to access **contextual information** from all other patches. This enables modeling of **long-range dependencies** across the image that would otherwise be lost when split into patches. Without MSA, the model would have no knowledge of spatial relationships. MSA computes **attention weights** for each pair of feature maps. These attention weights represent the importance of each feature map to the other, and they are used to compute a weighted sum of the feature maps.

### 2.5 Equation 3
Equation 3 involves coding the last two blocks in the Transformer Encoder: **Layer Normalization (LN)** and the **Multi-Layer Perceptron (MLP)** layer. The author states that the MLP contains **two linear layers** with a ```GELU (Gaussian Error Linear Units) non-linear activation function```. It also states that every linear layer in the MLP block has a **dropout layer** of value ```0.1```.  The MLP block is responsible for capturing complex, non-linear relationships within the local context of each token.


<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/b4a7957c-ff76-445f-9ff5-c3cc4bc7591b" width="60%"/>
</p>


The combination of linear transformations, non-linear activations, and dropout makes the model more flexible and expressive, helping it understand and handle different visual tasks effectively.

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/e9346ee1-4e4e-4705-be62-d377cf85c644" width="20%"/>
</p>

The hidden size of the MLP is ```3072``` and the input and output size of the MLP is equal to the embedding dimension ```768```. Above is a visual representation of the MLP and LN.

### 2.6 Equation 4

This expression denotes that for the last layer ```L```, the output ```y_L``` is obtained by applying a **LayerNorm (LN)** to the ```0``` index token of ```z_L```.

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/b6428b9e-c158-46ce-9422-1fafbadc4427" width="55%"/>
</p>

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/e101f5c5-bb44-4db5-b002-1bd0f7a40511" width="10%"/>
</p>




-----------------
<a name="transformer"></a>
## 3. Coding Transformers for Image Recognition: From Pixels to Predictions

### 3.1 Equation 1

We will start by coding equation 1 which is first to transform our input image into patches. 



<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/4e5658d7-366b-48b8-a982-4859296d0333" />
</p>


**Pseudocode:**

x_input = [class_token, patch_1, patch_2, ..., patch_N] + [class_token_pos, patch_1_pos, patch_2_pos, ..., patch_N_pos]

#### 3.1.1 Patching

There are 2 ways to turn our image into patches:

**1) Using raw image patches**

In the figure below, we looped through the different height and width dimensions of a single image and plotted individual patches. However, this method can be computationally expensive and time-consuming. It took ```16.780``` seconds to output the image below.

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/48e040df-339d-42ec-b032-b6bf87a4cf1e" width="50%"/>
</p>

**2) Using feature maps**

The author of the ViT paper proposed a **hybrid model** in which we can use the **feature maps** of a CNN. The patches can have a spatial dimension of ```1x1```, that is the input sequence is created by flattening the spatial dimensions of the feature map and then projecting it into the Transformer dimension.

We can use the convolutional operation with ```nn.Conv2d``` where we set the **kernel size** and **stride** equal to the **patch size**. That is, the convolutional kernel will be of size ```(patch size x patch size)```. The output will be a **learnable embedding** also called ```Linear Projection``` in the ViT paper. Recall our input is of size ```(224, 224, 3)``` and we want our output to be ```(196, 768)``` where ```768 = D = embedding dimension``` as in the table shown before. This means that **each** image will be embedded into a learnable vector of size ```768```.

```python
    nn.Conv2d(in_channels=3, out_channels=D, kernel_size=patch_size, stride=patch_size, padding=0)
```

Our output after the convolutional operation is of size ```(768, 14, 14) -> [embedding_dim, feature_map_height, feature_map_width]``` and below is an example of an input image and the first 10 **feature maps or activation maps**. These feature maps all represent our original image and they are the **learnable embedding** of our image.

| Input Image | Feature Maps 
|---------|---------|
| ![Image 2](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/cb387506-62a7-47cb-9471-ad571b75635e) | ![Image 3](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/b3504ddc-4843-4182-aca1-fd4bc7d0f2c3) |

Our patch embedding is still in ```2D``` and we need to transform it to ```1D``` similar to the token representation in NLP before projecting it in the Transformer model. Therefore, we need to transform our feature map from ```(768, 14, 14)``` to be ```(768, 196)```
 and then ```permute``` it to become ```(196, 768)```. We need to flatten the spatial dimensions of the feature map. Below is the representation of the first 10 feature maps flattened.


```python
    nn.Flatten(start_dim=2, end_dim=-1)
```

| Flattened Feature Maps |
|-------|
| ![Image 6](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/5833ab24-a355-4bdf-bd11-fbb2789b280c) |
| ![Image 7](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/b36c8c73-dd1c-44bf-bddc-0e10dde76aef) |
| ![Image 8](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/9e109b22-4eb5-4a0b-afcc-bf63c18d695f) |
| ![Image 9](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/f9458337-3657-4a64-88b5-1c7e7a79315a) |
| ![Image 10](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/4a20d1b4-6f4e-4418-9635-0bf9fda03bda) |
| ![Image 11](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/4996e64e-a42b-4d46-a19c-8b0af92a4c7a) |
| ![Image 12](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/ce9f2aa3-abfd-49ce-bcbe-a0736137e7db) |
| ![Image 13](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/a76236e2-ce65-4a84-86f3-4b8acd22e302) |
| ![Image 14](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/b37c230c-a773-49f9-8472-e1513ebc63df) |
| ![Image 15](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-in-Simulated-6DoF-Robot-Environments/assets/59663734/31c9ff7f-cc53-407a-bbcb-461fb6c49079) |

With this, we have turned our single ```2D``` image into a **1D learnable embedding vector** or ```Linear Projection of Flattned Patches```. 





#### 3.1.2 Class Token
Our extra learnable class embedding token should be of size ```(1 x 768)``` where ```768``` is the **embedding dimension**. We create a learnable embedding as shown below:

```python
class_token = nn.Parameter(torch.ones(batch_size, 1, 768), requires_grad=True)
```

We also need to **prepend** it to our **patch embedding**. We use ```torch.cat``` on the first dimension to do so. The patch embedding is of size ```(196, 768)``` and our output (with prepend class token) is of size ```(197 x 768)```.

#### 3.1.3 Position Embedding

Next, we create learnable 1D position embedding of size ```(197 x 768)``` using ```torch.rand``` specifying ```requires_grad=True``` to make it learnable. Note that we have ```197``` because we have to prepend the class token to the patch embedding.


```python
position_embedding = nn.Parameter(torch.ones(batch_size, num_patches+1, 768), requires_grad=True)
```

Next, we need to add our position embedding to the patch embedding with the prepend class token.

#### 3.1.4 Patch + Class Token + Position Embedding

We created a class that takes in an image, applies convolutional operation, flattens it, prepends a class token, and add a positional embedding:

```python
patchify = PatchClassPositionEmbedding(in_channels=3, embedding_dim=768, patch_size=16, num_patches=196, batch_size=1)
```

### 3.2 Equation 2
The next phase will be to implement the **Layer Normalization (LN)** and the **Multi-Head Self Attention (MSA)** layer. As explained above, the LN allows neural networks to optimize over data samples with similar distributions (similar mean and standard deviations) more easily than those with varying distributions and the MSA identifies the relation between the image patches to create a learned representation of an image.

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/8f3f1e95-1add-405b-b104-c009cc68993e" width="25%"/>
</p>

**Pseudocode:**

```python
x_output_MSA_block = MSA_layer(LN_layer(x_input)) + x_input
```

Note that adding the ```x_input``` after the MSA and LN is a **residual connection**. We will use the PyTorch implementation of the MSA and Layer Norm to code the first 2 blocks in the Transformer Encoder.

```python
multihead_attn_block = MultiHeadAttentionBlock(embed_dim=768, num_heads=12)
```

Note that we have a **residual connection** that adds the input back after the MSA block however, we will implement this later on.

### 3.3 Equation 3
Equation 3 contains a layer norm and an MLP block which consists of a Fully Connected layer followed by a non-linear GELU activation function, a dropout for regularization, a second linear transformation using a Fully Connected layer, and finally another dropout. Similarly to equation 2. we will create a class to implement it but will skip the skip connection for now (no pun intended).

**Pseudocode:**

```python
x_output_MLP_block = MLP_layer(LN_layer(x_output_MSA_block)) + x_output_MSA_block
```

<p align="center">
  <img src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/e9346ee1-4e4e-4705-be62-d377cf85c644" width="25%"/>
</p>

```python
mlp_block = MLPBlock(embedding_dim=768, mlp_size=3072, dropout=0.1)
```


### 3.4 Transformer Encoder
We have been skipping the skip connections tills now (again no pun intended). It is time to build the Transformer Encoder architecture using the MLP Block and Multi-Head Attention Block we designed. The input of the MSA Block (MSA + LN) is added back to the output of the MSA Block before it goes as input to the MLP Block (LN + MLP). Then again, the input of the MLP Block is added back to the output of the MLP Block. 


**Pseudocode:**

```python
x_input -> MSA_block -> [MSA_block_output + x_input] -> MLP_block -> [MLP_block_output + MSA_block_output + x_input
```

Note that we call the Multi-Head Attention block and MLP block that we already coded above and add their respective residual connections.

```python
    def forward(self, x):
        
        # residual connection for MSA Block
        x  = self.multihead_attn_block(x) + x
        
        # residual connection for MLP Block
        x = self.mlp_block(x) + x
        
        return x
```

### 3.5 Equation 4

For the last equation, we simply require a ```torch.nn.LayerNorm()``` layer and a ```torch.nn.Linear()``` layer to transform the **logit** outputs from the 0th index of the Transformer Encoder into the desired number of **target** classes.


```python
y = Linear_layer(LN_layer(x_output_MLP_block[0]))
```

```python
classification_head = nn.Sequential(nn.LayerNorm(normalized_shape=embed_dim),
                                    nn.Linear(in_features=embed_dim,
                                    out_features=num_class))
```

### 3.5 Custom Vision Transformer
Finally, we need to assemble all our code clocks in our custom Vision Transformer model. As in the ViT paper, we will use ```12``` Transformer Encoder blocks. We used an Adam optimizer with ```0.003``` learning rate and a weight decay of ```0.3```. For the loss function, we use the Cross Entropy.

```python
class VisualTransformerCustom(nn.Module):
    def __init__(self, 
                 img_size:int=224,
                 in_channels:int=3, 
                 embed_dim:int=768,
                 patch_size:int=16, 
                 num_layers:int=12,
                 num_heads: int = 12,
                 hidden_dim:int = 3072, 
                 dropout:float = 0.1,
                 embed_dropout:int=0.1,
                 num_class:int=3):
        
        super(VisualTransformerCustom, self).__init__()
        
        # Create an instance of patch embedding layer
        self.patchify = PatchClassPositionEmbedding(in_channels=in_channels, 
                                                    embed_dim=embed_dim, 
                                                    patch_size=patch_size, 
                                                    img_size=img_size,
                                                    embed_dropout=embed_dropout)
        
        # Create a stack of Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(embed_dim=embed_dim, num_heads=num_heads, hidden_dim=hidden_dim, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Create MLP Head
        self.classification_head = nn.Sequential(nn.LayerNorm(normalized_shape=embed_dim),
                                            nn.Linear(in_features=embed_dim,
                                            out_features=num_class))
        
    def forward(self, x):
        
        # Patchify
        x = self.patchify(x)
        
        # Pass through transformer encoder blocks
        for transformer_block in self.transformer_blocks:
            x = transformer_block(x)
        
        # put 0 index logit through classifier (get last layer of x)
        x = self.classification_head(x[:, 0])
        
        return x
```

```python
Epoch: 25 | train_loss: 1.0992 | train_acc: 0.3370 | test_loss: 1.0987 | test_acc: 0.3312
```


We trained our model for ```25``` epochs. Below are the accuracy and loss curves for the train and test dataset. We observe our test and train accuracy is around ```33%```, which is no better if we randomly select a class for an object (since we have 3 classes). Note from the confusion matrix that we are only predicting cardboard. 

But why is our custom Vision Transformer trained from scratch failing? 

- First, this is because the ViT model uses far more amount of data. They used three different datasets each with millions of images: **1.3M (ImageNet-1k)**, **14M (ImageNet-21k)**, **303M (JFT)** compared to our total ```150``` images.
- We trained our model for ```10``` epochs, however, the original ViT ```7``` epochs for the largest dataset and  ```90, 300``` epochs for ImageNet.
- They used a large batch size ```4096``` compared to our ```32```.
- They also employed **regularization** techniques such as **learning rate warmup**, **learning rate decay**, and **gradient clipping** to prevent overfitting. We didn't have any of those.

| Loss and Accuracy Curves | Confusion Matrix |
|---------|---------|
| ![Image 1](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/14f1429c-f0e4-47a2-a364-4fb437b743cf) | ![Image 2](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/663ef564-5220-45d6-80d4-4302e6933050) |

Now that it makes sense why our model failed, the next step will be to use the **transfer learning** approach. We will use a pre-trained ViT model to train on our small dataset.


### 3.6 Pretrained Vision Transformer
The next best solution will be to use a pre-trained ViT model from ```torchvision.models```. We first the pre-trained weights for **ViT-Base** trained on ```ImageNet-1k``` and then set up the ViT model instance via ```torchvision.models.vit_b_16```.  Note that we want to freeze the base parameters but change the classifier head with our own, that is, since we must classify 3 classes, the output shape must be 3 too.

Similarly, as before, we trained our model using an **Adam optimizer** with a learning rate of ```0.001``` and a **cross-entropy** loss function. Below is the result of the loss and accuracy for ```25``` epochs:

```python
Epoch: 25 | train_loss: 0.0194 | train_acc: 1.0000 | test_loss: 0.0304 | test_acc: 1.0000
```
We got perfect training and test accuracy scores of ```100%``` for both the training and test datasets. This can be further confirmed with the confusion matrix below:

| Loss and Accuracy Curves | Confusion Matrix |
|---------|---------|
| ![Image 1](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/d893da01-5fbf-4bf2-89af-d4ebd1bee1f4) | ![Image 2](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/56e1cff5-896e-4373-a042-beecbb263a7c) |

We also performed inference of some test samples and below you can see that for the first image, it was tricky because both the plane and the plate are the exact same color. Nevertheless, our model would successfully classify our object with ```59%``` accuracy. For the cardboard and vase samples, it was easy for our model to classify them with over ```99%``` accuracy.

||||
|---------|---------|---------|
| <img width="325" alt="image" src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/550e1df5-a2d8-424f-bb4c-05c83f8d3667"> | <img width="337" alt="image" src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/2f9e1a97-d557-49b5-839a-8cc688c573d0"> | <img width="353" alt="image" src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/7555462e-b705-4bd9-9bcd-e740711a9ffc"> |

-----------------




<a name="simulation"></a>
## 4. Digital Twin with Transformers
Now the most interesting part of the project is to be able to use our trained model in Unity. Running a Python script inside Unity can be a little problematic as we not only want to output the value of the AI model in the console but be able to use it and run a full simulation in the Game Scene. Below are some possible solutions: 

1. The [Python Scripting package](https://docs.unity3d.com/Packages/com.unity.scripting.python@7.0/manual/index.html) allows us to run our inference code but it works only in the Unity Editor and is not available in runtime builds. 

2. We can also use the [IronPython package](https://github.com/exodrifter/unity-python) package however, upon testing we got some dependency errors.

3. Another solution would be to convert our inference Python script to C# and then, run it as usual in Unity. However, we needed PyTorchSharp API for this one and some API has changed since as explained [here](https://github.com/dotnet/TorchSharp/issues/740).

4. So the one solution that worked was to output our AI model as an ONNX format and then run it using the [Barracuda](https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/GettingStarted.html) package.

### 4.1 Process Flow
Now that we can run our AI model in Unity we need to set up the whole workflow for our simulation. Below are the tasks which we would want to perform:

1. Randomly instantiate one of the objects.
2. Move the object on the main conveyor belt.
3. Stop at markers.
4. A camera takes a picture of the object.
5. The image goes to the AI model for inference.
6. The AI model classifies the object.
7. Based on the object, we instantiate the correct gripper for the robot.
8. The robot grabs the object and places it on the correct conveyor belt.
9. The loop goes on as such.

Below we build a flowchart of the different processes described above:

![ViT_workflow](https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/2d194722-f0b8-4104-9826-2b227cbe27b3)

### 4.2 Path Planning for Box
The grip to grab boxes is similar to the [Boston Dynamics' Strech robot](https://bostondynamics.com/products/stretch/). We have suction cups at the bottom which means the robot will need to pick up a box from the top. Upon picking, the robot will need to rotate its ```J1``` joint counter-clockwise and put the box on its left conveyor belt. We first needed to plan the trajectory of this motion as shown below:

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/85d9798d-44ad-4d6c-bd49-d01b2bd149b2" controls="controls" style="max-width: 730px;">
  </video>
</div>


### 4.3 Path Planning for Vase
For the vase we cannot use the same grip with suction cups as above, instead, we will use a grip similar to a forklift fork. The fork will come to seize the vase at its neck, the robot will then rotate clockwise and put the vase on the right conveyor belt. Similarly, as above, we plan the trajectory of the robot for this motion:

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/0d5994b5-a0a7-4212-990c-d45d79c36ce3" controls="controls" style="max-width: 730px;">
  </video>
</div>

Note that we do not have a third grip for the plate object as plate objects will continue on the main conveyor belt where they will fall in a wooden container.

### 4.4 Object Classification
As mentioned above, we place another camera just above the markers and take a picture of the object. The picture then goes to our ```ViT``` model which **classifies** the object's **class**. We then displace the picture of the object and the classification label on a panel in Unity as shown below:

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/cdd793a1-6ab7-46c4-9538-7486227d3f2b" controls="controls" style="max-width: 730px;">
  </video>
</div>


### 4.5 End-to-End Simulation
Finally, we put all the building blocks together and created the full end-to-end simulation for the palletization of boxes, plates, and, vases.

<div style="text-align: center;">
  <video src="https://github.com/yudhisteer/Vision-Transformer-Based-Multi-Class-Classification-for-Simulated-6DoF-Robot/assets/59663734/efd4281a-1cd8-4220-b5b9-bc7e7b89ea23" controls="controls" style="max-width: 730px;">
  </video>
</div>



----------
## Conclusion

The project demonstrates how to build your custom Vision Transformer model - although the one we created from scratch had poor accuracy, we used a pre-trained one - within Unity to classify objects and simulate the palletization process as done in the industry. By building a Digital Twin of the robot and the workspace, we save time and effort in testing the best scenario on how to optimize the palletizing process. We can perform path planning for the 6 DOF robot to avoid any collisions and calculate the least time required to do certain tasks. Note that we generated a synthetic dataset with Unity itself that gave us more control over the quality and quantity of images we wanted but also saved us manual effort to take the pictures ourselves which could take days to do so. From collection of data to training of a Vision Transformer and implementation in Unity, we built a full Digital Twin for a 6 DoF robotic arm doing palletization. 








## References
1. https://arxiv.org/abs/2010.11929
2. https://www.youtube.com/watch?v=TrdevFK_am4&ab_channel=YannicKilcher
3. https://www.youtube.com/watch?v=j3VNqtJUoz0&ab_channel=DeepFindr
4. https://www.youtube.com/watch?v=DVoHvmww2lQ&list=PLpZBeKTZRGPMddKHcsJAOIghV8MwzwQV6&index=1&ab_channel=AICoffeeBreakwithLetitia
5. https://www.youtube.com/watch?v=j6kuz_NqkG0&ab_channel=AleksaGordi%C4%87-TheAIEpiphany
6. https://www.youtube.com/watch?v=DVoHvmww2lQ&list=PLpZBeKTZRGPMddKHcsJAOIghV8MwzwQV6&ab_channel=AICoffeeBreakwithLetitia
7. https://github.com/mashaan14/VisionTransformer-MNIST/blob/main/VisionTransformer_MNIST.ipynb
8. https://towardsdatascience.com/a-comprehensive-guide-to-swin-transformer-64965f89d14c
9. https://www.youtube.com/watch?v=YAgjfMR9R_M&ab_channel=MichiganOnline
10. https://jalammar.github.io/illustrated-transformer/
11. https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/
12. https://docs.unity3d.com/Packages/com.unity.barracuda@3.0/manual/GettingStarted.html
13. https://github.com/cj-mills/unity-barracuda-inference-image-classification
