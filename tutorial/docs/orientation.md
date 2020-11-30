---
id: orientation
sidebar_label: 2. Orientation
title: Orientation Module
slug: /orientation-module
---

import CodeBlock from "../src/utils/CodeBlock"
import OutputBlock from "../src/utils/OutputBlock"

To predict the orientation of an aligned student-id image inputted from the detection module, we shall quickly develop an image classification model and train it on our [orientation dataset](https://github.com/MbassiJaphet/pytorch-for-information-extraction/tree/master/code/datasets/orientation). We expect the trained orientation model to predict the confidence scores for orientation angles (90, 180, 270, and 360) for an input image.

So, let's resolve the imports of our orientation module.
<CodeBlock lines={[5,6,17,19]} file="orientation_module_imports"></CodeBlock>

## **2.1. Orientation Dataset**
The [orientation dataset](https://github.com/MbassiJaphet/pytorch-for-information-extraction/tree/master/code/datasets/orientation) consist of folders containing four subfolders, whereby each subfolder is named according to one of the four orientation classes i.e. **'090'**, **'180'**, **'270'**, and **'360'**. Each subfolder contains images rotated according to their folder's name.

Pytorch provides [torchvision.datasets.ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder) for loading datasets with such format without requiring us to hardcode a custom dataset class for the data like we did for the detection dataset.
![img](../static/img/orientation-datasets.svg)


### 2.1.1. Define transforms for orientation datasets
Before instantiating our various orientation datasets, we have to define the various transforms which shall be used to initialize them.
<CodeBlock lines={[1,4,5,6,7,8,9,10,14,21]} file="orientation_dataset_transforms"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   Defined and initialized various transforms specific to each of our orientation datasets(training, validation, and testing). We did that by **importing** and leveraging [torchvision.transforms](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=torchvision%20transforms) which is a module containing common image transformations.
*   [transforms.Compose](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms%20compose#torchvision.transforms.Compose) composes several transforms together.
*   [transforms.Resize](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms%20resize#torchvision.transforms.Resize) resizes the input image to the given size.
*   [transforms.RandomAffine](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms%20randomaffine#torchvision.transforms.RandomAffine) randomly affines transformation of the image keeping center invariant.
*   [transforms.RandomApply](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms%20randomapply#torchvision.transforms.RandomApply) randomly a list of transformations with a given probability.
*   [transforms.RandomGrayscale](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms%20randomgrayscale#torchvision.transforms.RandomGrayscale) randomly convert image to grayscale with a probability of **'p'**.
*   [transforms.Normalize](https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=transforms%20normalize#torchvision.transforms.Normalize) normalize a tensor image with mean and standard deviation.
*   **Note** that we only applied data augmentation on the training dataset. This is so that our model can easily generalize input data.

</details>


### 2.1.2.  Instantiate orientation datasets
We shall leverage Pytorch inbuilt torchvision.datasets.ImageFolder class to effortlessly instantiate our orientation training, validation, and testing datasets.
<CodeBlock lines={[1,10,12,14]} file="orientation_dataset_init"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   We initialized **training**, **validation**, and **testing** datasets using [torchvision.datasets.ImageFolder](https://pytorch.org/docs/stable/torchvision/datasets.html#torchvision.datasets.ImageFolder) with their respective **folders**.
*   We initialized the variables``orientation_classes``, and ``num_orientation_classes`` to values of our **orientation classes** and their **number** respectively.

</details>

<p></p>

Just checking the names and number of classes from our orientation dataset to make sure everything is **OK**!
<CodeBlock lines={[7,8,10,11]} file="orientation_dataset_classes"></CodeBlock>
<OutputBlock file="orientation_dataset_classes_output"></OutputBlock>

### 2.1.3. Visualize orientation dataset
<CodeBlock lines={[1,7,8,10,11]} file="orientation_dataset_visualize"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   Randomly selected a group of four elements from our orientation training dataset as ``(image_tensor, label_tensor)`` pairs.
*   Denormalized ``image_tensor`` for each pair and had each image plotted displaying their corresponding classes.

</details>

![img](../static/img/orientation-sample.svg)


## **2.2. Orientation Model**
### 2.2.1. Define Orientation Model
**Note** that the model architecture defined below expects input image tensors of shape  **(3 x 224 x 224)** taking after transforms of the orientation datasets.
Let's define an architecture for our orientation model from scratch.
<CodeBlock lines={[1,2,4,8,9,10,13,14,17,18,20,22,24,25,26,28,30,31,32]} file="orientation_model"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   Defined ``OrientationModel`` extending [torch.nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch%20nn%20module#torch.nn.Module). The constructor argument ``num_classes`` is equivalent to desired number of **classes/labels**.
*   Defined the feed-forward behavior of the neural network by overriding the [``forward``](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch%20nn%20module%20forward#torch.nn.Module.forward) method.

</details>

Now that we have defined the architecture of our orientation model, let's define the helper function to instantiate it !
<CodeBlock lines={[4]} file="orientation_model_init_function"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   Used [orientation_model.load_state_dict()](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict) to set model weights from state dictionary if ``state_dict`` is given.

</details>

### 2.2.2. Specify checkpoint and instantiate the model
Looking forward to **resumable** training and saving of our orientation model, we shall now specify the checkpoints for the **state dictionaries** of the model and its training optimizer while initializing the model at once.
<CodeBlock lines={[2,14,22]} file="orientation_checkpoint"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   Selected available computational hardware using [torch.device()](https://pytorch.org/docs/stable/tensor_attributes.html?highlight=torch%20device#torch.torch.device).

</details>
<OutputBlock file="orientation_checkpoint_output"></OutputBlock>

Now let's print our orientation model to check if it has been initialized as we expect.
<CodeBlock file="orientation_model_visualize"></CodeBlock>
<OutputBlock file="orientation_model_visualize_output"></OutputBlock>

## **2.3. Training and Validation**
### 2.3.1. Specify data loaders
After initializing the various orientation datasets, let us use them to specify data loaders which shall be used for training, validation, and testing.
<CodeBlock lines={[4,8,12,17]} file="orientation_dataset_loaders"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   Initialized data loaders for each of our orientation datasets (training, validation and testing) by using [torch.utils.data.DataLoader()](https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data%20dataloader#torch.utils.data.DataLoader).
*   Initialized the dictionary variable '``orientation_loaders ``', which references all of the data loaders.

</details>

### 2.3.2. Define loss function and optimizer
Let's initialize the optimizer for training the orientation model, and get ready for training !
<CodeBlock lines={[2,4,6,11]} file="orientation_optimizer"></CodeBlock>

<OutputBlock file="orientation_optimizer_init_output"></OutputBlock>

### 2.3.3. Define training function
<CodeBlock lines={[4,21,35,36,38,41,42,43,44,45,47,49,53,54,58,59,61]} file="orientation_model_train_function"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   Moved the **model** to the computation device as the ``(data, target)`` pairs from ``loaders['train']``.
*   Within the training loop, we reset the gradients before predicting ``output`` for each ``data``, and its compute ``loss`` to its ``target``.
*   Find best ``loss`` as ``valid_loss`` and update checkpoint accordingly.


</details>


### 2.3.4. Train orientation model
Now let's train our orientation model for 20 epochs.
<CodeBlock file="orientation_model_train"></CodeBlock>
<OutputBlock file="orientation_model_train_output"></OutputBlock>

### 2.3.5. Resume training orientation model
At the end of every epoch, we had the checkpoints of the orientation module updated. Now let's use these updated checkpoints to reload the orientation model with orientation optimizer and resume the training up to **'30'** epochs.

:::important

To reload the orientation model and the orientation optimizer from the checkpoint, simply re-run the code cells in Section 2.2.2. and Section 2.3.2 respectively. Just make sure ``load_orientation_checkpoint`` is set to ``True``. The resulting outputs shall be identical to the ones below.

:::

Reloading orientation model from the checkpoint. (Section 2.2.2)
<CodeBlock lines={[2,4,14,22]} file="orientation_checkpoint"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   Loaded checkpoint using [torch.load()](https://pytorch.org/docs/stable/generated/torch.load.html?highlight=torch%20load#torch.load). The argument ``map_location`` is used to specify the computing device into which the checkpoint is loaded.

</details>

<OutputBlock file="orientation_model_init_checkpoint_output"></OutputBlock>

Reloading orientation optimizer from the checkpoint (Section 2.3.2)
<CodeBlock lines={[2,4,6,8,9,11]} file="orientation_optimizer"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   Used [orientation_optimizer.load_state_dict()](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict) to initialize optimizer weights if ``orientation_optimizer_state_dict`` is available. This sets the optimizer to  the state after that of the previous training.

</details>

<OutputBlock file="orientation_optimizer_init_checkpoint_output"></OutputBlock>

Now let's resume the training of our orientation model.
<CodeBlock file="orientation_model_train_resume"></CodeBlock>
<OutputBlock file="orientation_model_train_resume_output"></OutputBlock>

You notice that the training starts from **epoch 21** since the orientation model has already been trained for **20 epochs**.

### 2.3.6. Evaluate orientation model
To conclude on the performance of your models, it is always of good practice to evaluate them on sample data. We shall evaluate the performance of the orientation model on sample images from the testing dataset.

But, before that let's define the test function.
<CodeBlock lines={[2,5,12,13,14,16,18,20,22,24,26,27]} file="orientation_model_test_function"></CodeBlock>
<details open><summary>Code Brieffings</summary>

*   Put the model to evaluation mode using [model.eval()](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch%20nn%20module%20eval#torch.nn.Module.eval). This disables some training behaviors of our model such as **batch normalization** and dropout layers.
*   Iterate **batches** of the **orientation test loader** for ``(data, target)`` pairs.
*   Move ``(data, target)`` pairs to computation **device/hardware** using [to()](https://pytorch.org/docs/stable/tensors.html?highlight=torch%20tensor#torch.Tensor.to) method.
*   Predict ``output`` for ``data`` and compute ``loss`` to targets. Then the average total loss is computed as``test_loss``.

</details>

With our test function defined, we shall now use it to evaluate the performance of the orientation model on the orientation test dataset.
<CodeBlock file="orientation_model_test"></CodeBlock>
<OutputBlock file="orientation_model_test_output"></OutputBlock>

## 2.4 Orientation Correction
Let's properly visualize the performance of our orientation model via inference on sample images from the test dataset one at a time.

Keep in mind that the objective behind an orientation module is to detect the orientation of an aligned document image, and to rectify it where necessary. Therefore, after inferencing every single image, we have shall apply the proper transformation to the image to rectify its orientation if necessary.
<CodeBlock lines={[8,9,10,14,16]} file="orientation_model_prediction_visualize"></CodeBlock>

![img](../static/img/orientation-prediction.svg)

<details><summary>More Outputs</summary>

![img](../static/img/orientation-prediction-1.svg)
<p></p>

![img](../static/img/orientation-prediction-2.svg)

</details>
