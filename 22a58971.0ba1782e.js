(window.webpackJsonp=window.webpackJsonp||[]).push([[5],{326:function(e,t,n){"use strict";n.r(t),t.default=n.p+"assets/images/detection-datasets-ac8430fb8f4be3539a0ad271ce519111.svg"},327:function(e,t,n){"use strict";n.r(t),t.default=n.p+"assets/images/detection-sample-2297f873c30fffa91d665d132959c6b2.svg"},328:function(e,t,n){"use strict";n.r(t),t.default=n.p+"assets/images/student-id-01-5375f7492350b2bca5e95746be434e85.svg"},329:function(e,t,n){"use strict";n.r(t),t.default=n.p+"assets/images/detection-prediction-3858b74e87735a5587fe77b5c146f5db.svg"},330:function(e,t,n){"use strict";n.r(t),t.default=n.p+"assets/images/detection-prediction-1-9c3a1f985f13d2b285adc339b53b6512.svg"},331:function(e,t,n){"use strict";n.r(t),t.default=n.p+"assets/images/detection-prediction-2-312c4b46db7819d52b6526af1c77c6cc.svg"},332:function(e,t,n){"use strict";n.r(t),t.default=n.p+"assets/images/image-alignment-fa0084d9b2299b6f93223064468e8933.svg"},60:function(e,t,n){"use strict";n.r(t),n.d(t,"frontMatter",(function(){return l})),n.d(t,"metadata",(function(){return s})),n.d(t,"rightToc",(function(){return d})),n.d(t,"default",(function(){return p}));var i=n(2),a=n(6),o=(n(0),n(73)),r=(n(77),n(83)),c=n(80),l={id:"detection",sidebar_label:"1. Detection",title:"Detection Module",slug:"/detection-module"},s={unversionedId:"detection",id:"detection",isDocsHomePage:!1,title:"Detection Module",description:"To locate student-id(s) within images, we gonna leverage transfer learning via fine-tuning the state of art object segmentation algorithm Mask R-CNN  backboned by pre-trained ResNet-50 available in torchvision models gallery.",source:"@site/docs/detection.md",slug:"/detection-module",permalink:"/pytorch-for-information-extraction/detection-module",editUrl:"https://github.com/MbassiJaphet/pytorch-for-information-extraction/edit/master/docs/detection.md",version:"current",sidebar_label:"1. Detection",sidebar:"tutorial",previous:{title:"Getting Started",permalink:"/pytorch-for-information-extraction/introduction"},next:{title:"Orientation Module",permalink:"/pytorch-for-information-extraction/orientation-module"}},d=[{value:"<strong>1.1. Detection Dataset</strong>",id:"11-detection-dataset",children:[{value:"1.1.1. Define dataset class",id:"111-define-dataset-class",children:[]},{value:"1.1.2. Define transforms for detection dataset",id:"112-define-transforms-for-detection-dataset",children:[]},{value:"1.1.3. Instantiate detection datasets",id:"113-instantiate-detection-datasets",children:[]},{value:"1.1.4. Visualize detection dataset",id:"114-visualize-detection-dataset",children:[]}]},{value:"<strong>1.2. Detection Model</strong>",id:"12-detection-model",children:[{value:"1.2.1. Define detection model",id:"121-define-detection-model",children:[]},{value:"1.2.2. Specify checkpoints and instantiate the model",id:"122-specify-checkpoints-and-instantiate-the-model",children:[]}]},{value:"<strong>1.3. Training and Evaluation</strong>",id:"13-training-and-evaluation",children:[{value:"1.3.1. Specify data loaders",id:"131-specify-data-loaders",children:[]},{value:"1.3.2. Initialize optimizer",id:"132-initialize-optimizer",children:[]},{value:"1.3.3. Define training function",id:"133-define-training-function",children:[]},{value:"1.3.4 Train detection model",id:"134-train-detection-model",children:[]},{value:"1.3.5. Resume training detection model",id:"135-resume-training-detection-model",children:[]},{value:"1.3.6. Evaluate the detection model",id:"136-evaluate-the-detection-model",children:[]}]},{value:"<strong>1.4. Student ID Alignment</strong>",id:"14-student-id-alignment",children:[]}],b={rightToc:d};function p(e){var t=e.components,l=Object(a.a)(e,["components"]);return Object(o.b)("wrapper",Object(i.a)({},b,l,{components:t,mdxType:"MDXLayout"}),Object(o.b)("p",null,"To locate student-id(s) within images, we gonna leverage transfer learning via fine-tuning the state of art object segmentation algorithm ",Object(o.b)("a",Object(i.a)({parentName:"p"},{href:"https://arxiv.org/abs/1703.06870"}),Object(o.b)("strong",{parentName:"a"},"Mask R-CNN")),"  backboned by pre-trained ",Object(o.b)("a",Object(i.a)({parentName:"p"},{href:"https://pytorch.org/docs/stable/torchvision/models.html#mask-r-cnn"}),Object(o.b)("strong",{parentName:"a"},"ResNet-50"))," available in torchvision models gallery."),Object(o.b)("p",null,"So, let's resolve the imports of our detection module."),Object(o.b)(r.a,{lines:[5,6,17,19],file:"detection_module_imports",mdxType:"CodeBlock"}),Object(o.b)("details",{open:!0},Object(o.b)("summary",null,"Code Brieffing"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"Import ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/torch.html"}),"torch")," and ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/torchvision/"}),"torchvision")," which are libraries of the Pytorch project."),Object(o.b)("li",{parentName:"ul"},Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=torchvision%20transforms%20totensor#torchvision.transforms.ToTensor"}),"torchvision.transforms.ToTensor()")," reurns a function which takes in a PIL image and converts it to a ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/generated/torch.tensor.html?highlight=tensor#torch.tensor"}),"tensor"),"."),Object(o.b)("li",{parentName:"ul"},Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=torchvision%20transforms%20topilimage#torchvision.transforms.ToPILImage"}),"torchvision.transforms.ToPILImage()")," reurns a function which that does the opposite."))),Object(o.b)("h2",{id:"11-detection-dataset"},Object(o.b)("strong",{parentName:"h2"},"1.1. Detection Dataset")),Object(o.b)("h3",{id:"111-define-dataset-class"},"1.1.1. Define dataset class"),Object(o.b)("p",null,"A crucial requirement when fine-tuning, training, or inferencing models in Pytorch is to know the exact formats of data that specific models expect as inputs and compute as outputs."),Object(o.b)("p",null,"The input to the model is expected to be a list of tensors, each of shape ",Object(o.b)("inlineCode",{parentName:"p"},"[C, H, W]"),", one for each image, and should be in the range",Object(o.b)("inlineCode",{parentName:"p"},"0-1")," . Different images can have different sizes."),Object(o.b)("p",null,Object(o.b)("strong",{parentName:"p"},"Let's take a look at the format of targets expected by the model"),"."),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"boxes (",Object(o.b)("inlineCode",{parentName:"li"},"FloatTensor[N, 4]"),"): the ground-truth boxes in ",Object(o.b)("inlineCode",{parentName:"li"},"[x1, y1, x2, y2]")," format, with values of ",Object(o.b)("inlineCode",{parentName:"li"},"x")," between ",Object(o.b)("inlineCode",{parentName:"li"},"0")," and ",Object(o.b)("inlineCode",{parentName:"li"},"W")," and values of y between ",Object(o.b)("inlineCode",{parentName:"li"},"0")," and ",Object(o.b)("inlineCode",{parentName:"li"},"H"),"."),Object(o.b)("li",{parentName:"ul"},"labels (",Object(o.b)("inlineCode",{parentName:"li"},"Int64Tensor[N]"),"): the class label for each ground-truth box."),Object(o.b)("li",{parentName:"ul"},"masks (",Object(o.b)("inlineCode",{parentName:"li"},"UInt8Tensor[N, H, W]"),"): the segmentation binary masks for each instance.")),Object(o.b)("p",null,Object(o.b)("strong",{parentName:"p"},"Then, we shall also take a look at the format of outputs predicted by the model"),"."),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"boxes (",Object(o.b)("inlineCode",{parentName:"li"},"FloatTensor[N, 4]"),"): the predicted boxes in ",Object(o.b)("inlineCode",{parentName:"li"},"[x1, y1, x2, y2]")," format, with values of ",Object(o.b)("inlineCode",{parentName:"li"},"x")," between ",Object(o.b)("inlineCode",{parentName:"li"},"0")," and ",Object(o.b)("inlineCode",{parentName:"li"},"W")," and values of ",Object(o.b)("inlineCode",{parentName:"li"},"y")," between ",Object(o.b)("inlineCode",{parentName:"li"},"0")," and ",Object(o.b)("inlineCode",{parentName:"li"},"H"),"."),Object(o.b)("li",{parentName:"ul"},"labels (",Object(o.b)("inlineCode",{parentName:"li"},"Int64Tensor[N]"),"): the predicted labels for each image."),Object(o.b)("li",{parentName:"ul"},"scores (",Object(o.b)("inlineCode",{parentName:"li"},"Tensor[N]"),"): the scores or each prediction."),Object(o.b)("li",{parentName:"ul"},"masks (",Object(o.b)("inlineCode",{parentName:"li"},"UInt8Tensor[N, 1, H, W]"),"): the predicted masks for each instance, in the range ",Object(o.b)("inlineCode",{parentName:"li"},"0-1"),". To obtain the final segmentation masks, the soft masks can be thresholded, generally with a value of ",Object(o.b)("inlineCode",{parentName:"li"},"0.5")," (",Object(o.b)("inlineCode",{parentName:"li"},"mask >= 0.5"),").")),Object(o.b)("p",null,"Recall from the ",Object(o.b)("a",Object(i.a)({parentName:"p"},{href:"/introduction/#project-description/"}),"project description")," that we shall train our detection model on the ",Object(o.b)("a",Object(i.a)({parentName:"p"},{href:"https://github.com/MbassiJaphet/pytorch-for-information-extraction/tree/master/code/datasets/detection"}),Object(o.b)("strong",{parentName:"a"},"Student-ID"))," dataset. So let\u2019s examine its format !\n",Object(o.b)("img",{alt:"img",src:n(326).default})),Object(o.b)("p",null,"Now, knowing the formats of the Student-ID dataset as well as the formats of inputs/targets/outputs of the pre-trained model, we can confidently code a custom dataset class inheriting from ",Object(o.b)("a",Object(i.a)({parentName:"p"},{href:"https://pytorch.org/docs/stable/data.html#torch.utils.data.Dataset"}),"torch.utils.data.Dataset"),"."),Object(o.b)(r.a,{lines:[7,49,50,51,53,58,59,60,61,62,63,67],file:"detection_dataset",mdxType:"CodeBlock"}),Object(o.b)("details",{open:!0},Object(o.b)("summary",null,"Code Brieffing"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"We defined the ",Object(o.b)("inlineCode",{parentName:"li"},"DetectionDataset")," class initialized with ",Object(o.b)("inlineCode",{parentName:"li"},"data_path"),"(folder containing detection dataset), a ",Object(o.b)("inlineCode",{parentName:"li"},"mode"),"(",Object(o.b)("strong",{parentName:"li"},"'TRAIN'"),", ",Object(o.b)("strong",{parentName:"li"},"'VALID'"),", ",Object(o.b)("strong",{parentName:"li"},"'TEST'"),"), and ",Object(o.b)("inlineCode",{parentName:"li"},"transform")," (data augmentation function)."),Object(o.b)("li",{parentName:"ul"},"We implicitly assigned anything but our ",Object(o.b)("inlineCode",{parentName:"li"},"classes")," to the ",Object(o.b)("strong",{parentName:"li"},"'BACKGROUND'")," class."),Object(o.b)("li",{parentName:"ul"},"We implemented ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data%20dataset#torch.utils.data.Dataset"}),Object(o.b)("inlineCode",{parentName:"a"},"__getitem__"))," to return individual elements of our dataset as (",Object(o.b)("inlineCode",{parentName:"li"},"image_tensor"),", ",Object(o.b)("inlineCode",{parentName:"li"},"targets"),") pairs."),Object(o.b)("li",{parentName:"ul"},Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/generated/torch.from_numpy.html?highlight=torch%20from_numpy#torch.from_numpy"}),"torch.from_numpy()")," Creates a Tensor from a numpy.ndarray"),Object(o.b)("li",{parentName:"ul"},Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/generated/torch.as_tensor.html?highlight=torch%20as_tensor#torch.as_tensor"}),"torch.as_tensor()")," Convert the data into a torch.Tensor"))),Object(o.b)("h3",{id:"112-define-transforms-for-detection-dataset"},"1.1.2. Define transforms for detection dataset"),Object(o.b)("p",null,"Let's write some helper functions for data augmentation."),Object(o.b)(r.a,{file:"detection_dataset_transforms",mdxType:"CodeBlock"}),Object(o.b)("h3",{id:"113-instantiate-detection-datasets"},"1.1.3. Instantiate detection datasets"),Object(o.b)(r.a,{lines:[3,5,7],file:"detection_dataset_init",mdxType:"CodeBlock"}),Object(o.b)("details",{open:!0},Object(o.b)("summary",null,"Code Brieffings"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"We initialized ",Object(o.b)("strong",{parentName:"li"},"training"),", ",Object(o.b)("strong",{parentName:"li"},"validation"),", and ",Object(o.b)("strong",{parentName:"li"},"testing")," datasets using the modes 'TRAIN', 'VALID' and 'TEST' respectively."),Object(o.b)("li",{parentName:"ul"},"We ",Object(o.b)("strong",{parentName:"li"},"disabled")," data augmentation for testing dataset."),Object(o.b)("li",{parentName:"ul"},"We initialized the variables",Object(o.b)("inlineCode",{parentName:"li"},"detection_classes"),", and ",Object(o.b)("inlineCode",{parentName:"li"},"num_detection_classes")," to values of our ",Object(o.b)("strong",{parentName:"li"},"detection classes")," and their ",Object(o.b)("strong",{parentName:"li"},"number")," respectively."))),Object(o.b)("p",null),Object(o.b)("p",null,"Just checking the names and number of classes from our detection dataset to make sure everything is ",Object(o.b)("strong",{parentName:"p"},"OK"),"!"),Object(o.b)(r.a,{file:"detection_dataset_classes",mdxType:"CodeBlock"}),Object(o.b)(c.a,{file:"detection_dataset_classes_output",mdxType:"OutputBlock"}),Object(o.b)("h3",{id:"114-visualize-detection-dataset"},"1.1.4. Visualize detection dataset"),Object(o.b)(r.a,{lines:[6,8,9,20,21,24,25,29],file:"detection_dataset_visualize",mdxType:"CodeBlock"}),Object(o.b)("details",{open:!0},Object(o.b)("summary",null,"Code Brieffings"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"We selected an inidividual element from ",Object(o.b)("inlineCode",{parentName:"li"},"detection_train_set")," using ",Object(o.b)("inlineCode",{parentName:"li"},"id")," as (",Object(o.b)("inlineCode",{parentName:"li"},"image_tensor"),", ",Object(o.b)("inlineCode",{parentName:"li"},"targets"),") pairs."),Object(o.b)("li",{parentName:"ul"},"We retrieved bounding boxes, segmentation masks, and labels from the ",Object(o.b)("inlineCode",{parentName:"li"},"targets")," dictionary."),Object(o.b)("li",{parentName:"ul"},Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/generated/torch.zeros_like.html?highlight=torch%20zeros_like#torch.zeros_like"}),"torch.zeros_like()")," returns a tensor filled ",Object(o.b)("inlineCode",{parentName:"li"},"0s"),", with the same size as input."),Object(o.b)("li",{parentName:"ul"},Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/tensors.html?highlight=tensor%20item#torch.Tensor.item"}),"torch.Tensor.item()")," returns the value of this tensor as a standard Python number."))),Object(o.b)("p",null,Object(o.b)("img",{alt:"img",src:n(327).default})),Object(o.b)("h2",{id:"12-detection-model"},Object(o.b)("strong",{parentName:"h2"},"1.2. Detection Model")),Object(o.b)("h3",{id:"121-define-detection-model"},"1.2.1. Define detection model"),Object(o.b)("p",null,"Let's define a helper function to instantiate the detection model !"),Object(o.b)(r.a,{lines:[1,2,6,8,11,13,16,17],file:"detection_model_init_function",mdxType:"CodeBlock"}),Object(o.b)("details",{open:!0},Object(o.b)("summary",null,"Code Brieffings"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"We imported ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/torchvision/models.html?highlight=torchvision%20models%20detection%20faster_rcnn#mask-r-cnn"}),"Mask-RCNN predictor")," and ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/torchvision/models.html?highlight=torchvision%20models%20detection%20faster_rcnn#faster-r-cnn"}),"Fast-RCNN predictor")," heads."),Object(o.b)("li",{parentName:"ul"},"Loaded ",Object(o.b)("strong",{parentName:"li"},"Mask R-CNN model")," with pre-trained ",Object(o.b)("strong",{parentName:"li"},"ResNet-50-FPN")," backbone and ",Object(o.b)("strong",{parentName:"li"},"finetuned")," it using ",Object(o.b)("inlineCode",{parentName:"li"},"num_classes"),". Using the pre-trained model implicitly makes us use ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://en.wikipedia.org/wiki/Transfer_learning"}),Object(o.b)("strong",{parentName:"a"},"transfer learning"))," which in turn makes our model converge faster."),Object(o.b)("li",{parentName:"ul"},"Used ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict"}),"detection_model.load_state_dict()")," to set model weights from state dictionary if ",Object(o.b)("inlineCode",{parentName:"li"},"state_dict")," is given."))),Object(o.b)("blockquote",null,Object(o.b)("p",{parentName:"blockquote"},Object(o.b)("strong",{parentName:"p"},"Remark:")," The helper function above allows us to fine-tune the pre-trained ",Object(o.b)("strong",{parentName:"p"},"FastRCNNPredictor")," and ",Object(o.b)("strong",{parentName:"p"},"MaskRCNNPredictor")," with the desired number of classes, which are ",Object(o.b)("strong",{parentName:"p"},"'2'")," in our case i.e. for the 'BACKGROUND' and 'Student_ID' classes. The function also sets the number of hidden layers of ",Object(o.b)("strong",{parentName:"p"},"MaskRCNNPredictor")," to ",Object(o.b)("strong",{parentName:"p"},"'256'")," but we can decide to tweak that for the best of our model performance.")),Object(o.b)("h3",{id:"122-specify-checkpoints-and-instantiate-the-model"},"1.2.2. Specify checkpoints and instantiate the model"),Object(o.b)("p",null,"Looking forward to ",Object(o.b)("strong",{parentName:"p"},"resumable")," training and saving of our detection model, we shall now specify the checkpoints for the ",Object(o.b)("strong",{parentName:"p"},"state dictionaries")," for both the model and its training optimizer."),Object(o.b)(r.a,{lines:[2,14,22],file:"detection_checkpoint",mdxType:"CodeBlock"}),Object(o.b)("details",{open:!0},Object(o.b)("summary",null,"Code Brieffings"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"Selected available computational hardware using ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/tensor_attributes.html?highlight=torch%20device#torch.torch.device"}),"torch.device()"),"."),Object(o.b)("li",{parentName:"ul"},Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/cuda.html?highlight=torch%20cuda%20is_available#torch.cuda.is_available"}),"torch.cuda.is_available()")," returns ",Object(o.b)("inlineCode",{parentName:"li"},"True")," if cuda capable hardware(s) is/are found."),Object(o.b)("li",{parentName:"ul"},"Loaded checkpoints using ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/generated/torch.load.html?highlight=torch%20load#torch.load"}),"torch.load()"),". The argument ",Object(o.b)("inlineCode",{parentName:"li"},"map_location")," is used to specify the computing device into which the checkpoint is loaded. This very useful if you have no idea of the device type for which a tensor has been saved."))),Object(o.b)(c.a,{file:"detection_checkpoint_output",mdxType:"OutputBlock"}),Object(o.b)("h2",{id:"13-training-and-evaluation"},Object(o.b)("strong",{parentName:"h2"},"1.3. Training and Evaluation")),Object(o.b)("p",null,Object(o.b)("strong",{parentName:"p"},"Note")," that the files used for training and validation of detection module found ",Object(o.b)("inlineCode",{parentName:"p"},"./modules/detection/scripts")," folder were directly copied along with their dependencies from torchvision reference detection training scripts repository."),Object(o.b)("h3",{id:"131-specify-data-loaders"},"1.3.1. Specify data loaders"),Object(o.b)("p",null,"After initializing the various detection datasets, let us use them to specify data loaders which shall be used for training, validation, and testing."),Object(o.b)(r.a,{lines:[4,9,14,19],file:"detection_dataset_loaders",mdxType:"CodeBlock"}),Object(o.b)("details",{open:!0},Object(o.b)("summary",null,"Code Brieffings"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"Initialized data loaders for each of our detection datasets (training, validation and testing) by using ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/data.html?highlight=torch%20utils%20data%20dataloader#torch.utils.data.DataLoader"}),"torch.utils.data.DataLoader()"),"."),Object(o.b)("li",{parentName:"ul"},"Initialized the dictionary variable '",Object(o.b)("inlineCode",{parentName:"li"},"detection_loaders "),"', which references all of the data loaders."))),Object(o.b)("h3",{id:"132-initialize-optimizer"},"1.3.2. Initialize optimizer"),Object(o.b)("p",null,"Let's initialize the optimizer for training the detection model, and get ready for training !"),Object(o.b)(r.a,{lines:[2,4,,6,11],file:"detection_optimizer_init",mdxType:"CodeBlock"}),Object(o.b)(c.a,{file:"detection_optimizer_init_output",mdxType:"OutputBlock"}),Object(o.b)("h3",{id:"133-define-training-function"},"1.3.3. Define training function"),Object(o.b)("p",null,"Now, let's write the function that will train and validate our model for us. Inside the training function, we shall add a few lines of code that will save our model checkpoints."),Object(o.b)(r.a,{lines:[12,18,20,22],file:"detection_model_train_function",mdxType:"CodeBlock"}),Object(o.b)("h3",{id:"134-train-detection-model"},"1.3.4 Train detection model"),Object(o.b)("p",null,"So let\u2019s train our detection model for 20 epochs saving it at the end of each epoch."),Object(o.b)(r.a,{file:"detection_model_train",mdxType:"CodeBlock"}),Object(o.b)(c.a,{file:"detection_model_train_output",mdxType:"OutputBlock"}),Object(o.b)("h3",{id:"135-resume-training-detection-model"},"1.3.5. Resume training detection model"),Object(o.b)("p",null,"At the end of every epoch, we had the checkpoints of the detection module updated. Now let's use these updated checkpoints to reload the detection model and resume its training up to ",Object(o.b)("strong",{parentName:"p"},"'30'")," epochs."),Object(o.b)("div",{className:"admonition admonition-important alert alert--info"},Object(o.b)("div",Object(i.a)({parentName:"div"},{className:"admonition-heading"}),Object(o.b)("h5",{parentName:"div"},Object(o.b)("span",Object(i.a)({parentName:"h5"},{className:"admonition-icon"}),Object(o.b)("svg",Object(i.a)({parentName:"span"},{xmlns:"http://www.w3.org/2000/svg",width:"14",height:"16",viewBox:"0 0 14 16"}),Object(o.b)("path",Object(i.a)({parentName:"svg"},{fillRule:"evenodd",d:"M7 2.3c3.14 0 5.7 2.56 5.7 5.7s-2.56 5.7-5.7 5.7A5.71 5.71 0 0 1 1.3 8c0-3.14 2.56-5.7 5.7-5.7zM7 1C3.14 1 0 4.14 0 8s3.14 7 7 7 7-3.14 7-7-3.14-7-7-7zm1 3H6v5h2V4zm0 6H6v2h2v-2z"})))),"important")),Object(o.b)("div",Object(i.a)({parentName:"div"},{className:"admonition-content"}),Object(o.b)("p",{parentName:"div"},"To reload the detection model and the detection optimizer from the checkpoint, simply re-run the code cells in Section 1.2.2. and Section 1.3.2 respectively. Just make sure ",Object(o.b)("inlineCode",{parentName:"p"},"load_detection_checkpoint")," is set to ",Object(o.b)("inlineCode",{parentName:"p"},"True"),". The resulting outputs shall be identical to the ones below."))),Object(o.b)("p",null,"Reloading detection model from the checkpoint. (Section 1.2.2)"),Object(o.b)(r.a,{lines:[2,4,14,22],file:"detection_checkpoint",mdxType:"CodeBlock"}),Object(o.b)("details",{open:!0},Object(o.b)("summary",null,"Code Brieffings"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"Loaded checkpoint using ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/generated/torch.load.html?highlight=torch%20load#torch.load"}),"torch.load()"),". The argument ",Object(o.b)("inlineCode",{parentName:"li"},"map_location")," is used to specify the computing device into which the checkpoint is loaded."))),Object(o.b)(c.a,{file:"detection_model_init_checkpoint_output",mdxType:"OutputBlock"}),Object(o.b)("p",null,"Reloading detection optimizer from the checkpoint (Section 1.3.2)"),Object(o.b)(r.a,{lines:[2,4,,6,11],file:"detection_optimizer_init",mdxType:"CodeBlock"}),Object(o.b)("details",{open:!0},Object(o.b)("summary",null,"Code Brieffings"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"Used ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=load_state_dict#torch.nn.Module.load_state_dict"}),"detection_optimizer.load_state_dict()")," to initialize optimizer weights if ",Object(o.b)("inlineCode",{parentName:"li"},"detection_optimizer_state_dict")," is available. This sets the optimizer to  the state after that of the previous training."))),Object(o.b)(c.a,{file:"detection_optimizer_init_checkpoint_output",mdxType:"OutputBlock"}),Object(o.b)("p",null,"Now let's resume training of our detection model."),Object(o.b)(r.a,{file:"detection_model_train_resume",mdxType:"CodeBlock"}),Object(o.b)(c.a,{file:"detection_model_train_resume_output",mdxType:"OutputBlock"}),Object(o.b)("p",null,"You notice that the training start from epoch ",Object(o.b)("strong",{parentName:"p"},"21")," since the detection model has already been trained for 20 epochs."),Object(o.b)("h3",{id:"136-evaluate-the-detection-model"},"1.3.6. Evaluate the detection model"),Object(o.b)("p",null,"To conclude on the performance of your models, it is always of good practice to evaluate them on sample data. We shall evaluate the performance of the detection model on sample images from the testing dataset."),Object(o.b)("p",null,"Firstly, let's use our detection model to compute predictions for an input image from the test detection dataset."),Object(o.b)(r.a,{lines:[3,4,12,13,15],file:"detection_model_predict",mdxType:"CodeBlock"}),Object(o.b)("details",{open:!0},Object(o.b)("summary",null,"Code Brieffings"),Object(o.b)("ul",null,Object(o.b)("li",{parentName:"ul"},"Selected an image URL from the testing dataset for inference."),Object(o.b)("li",{parentName:"ul"},"Then put the model to evaluation mode using ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=torch%20nn%20module%20eval#torch.nn.Module.eval"}),"detection_model.eval()"),". That disables training features like batch normalization and dropout making inference faster."),Object(o.b)("li",{parentName:"ul"},"Disabled gradient calculations for operations on tensors ",Object(o.b)("strong",{parentName:"li"},"within a block")," using ",Object(o.b)("a",Object(i.a)({parentName:"li"},{href:"https://pytorch.org/docs/stable/generated/torch.no_grad.html?highlight=torch%20no_grad#torch.no_grad"}),Object(o.b)("inlineCode",{parentName:"a"},"with torch.no_grad():")),"."))),Object(o.b)("p",null,Object(o.b)("img",{alt:"img",src:n(328).default})),Object(o.b)("p",null,"Secondly, let's take a look at the raw outputs predicted by our detection model for the image above."),Object(o.b)(r.a,{file:"detection_model_predictions_raw",mdxType:"CodeBlock"}),Object(o.b)(c.a,{file:"detection_model_predictions_raw_output",mdxType:"OutputBlock"}),Object(o.b)("p",null,"As we can see the predictions are simply a dictionary containing ",Object(o.b)("strong",{parentName:"p"},"labels"),", ",Object(o.b)("strong",{parentName:"p"},"scores"),", ",Object(o.b)("strong",{parentName:"p"},"boxes"),", and ",Object(o.b)("strong",{parentName:"p"},"masks")," of detected objects in tensor format."),Object(o.b)("p",null),Object(o.b)("p",null,"Lastly, let's convert the raw predicted outputs into a human-understandable format for proper visualization."),Object(o.b)(r.a,{lines:[6,7,12,13,14,17,21],file:"detection_model_predictions_visualize",mdxType:"CodeBlock"}),Object(o.b)("p",null,Object(o.b)("img",{alt:"img",src:n(329).default})),Object(o.b)("details",null,Object(o.b)("summary",null,"More Outputs"),Object(o.b)("p",null,Object(o.b)("img",{alt:"img",src:n(330).default}),"\n",Object(o.b)("img",{alt:"img",src:n(331).default}))),Object(o.b)("h2",{id:"14-student-id-alignment"},Object(o.b)("strong",{parentName:"h2"},"1.4. Student ID Alignment")),Object(o.b)("p",null,"At this point, what is left to be done in this module is to align student-id(s) detected by out detection model. The aligned student-id(s) shall then be fed as input to the orientation module."),Object(o.b)(r.a,{lines:[2],file:"detection_module_image_alignment",mdxType:"CodeBlock"}),Object(o.b)("p",null,Object(o.b)("img",{alt:"img",src:n(332).default}),"\nNow, let's save our aligned student-id."),Object(o.b)(r.a,{lines:[2],file:"detection_module_image_alignment_save",mdxType:"CodeBlock"}))}p.isMDXComponent=!0},80:function(e,t,n){"use strict";var i=n(3),a=n(0),o=n.n(a),r=function(e){function t(t){var n;return(n=e.call(this,t)||this)._currentFile=null,n.state={outputString:""},n}Object(i.a)(t,e),t.getDerivedStateFromProps=function(e,t){return e.id!==t.prevFile?{outputString:"",prevFile:e.file}:null};var n=t.prototype;return n.componentDidMount=function(){this._loadAsyncData(this.props.file)},n.componentDidUpdate=function(e,t){this.state.outputString||this._loadAsyncData(this.props.file)},n.componentWillUnmount=function(){this._currentFile=null},n.render=function(){return o.a.createElement("pre",{className:"output-block"},o.a.createElement("code",null,this.state.outputString))},n._loadAsyncData=function(e){var t=this;this._currentFile=e,fetch("/pytorch-for-information-extraction/code-snippets/"+e+".txt").then((function(e){return e.text()})).then((function(n){e===t._currentFile&&t.setState({outputString:n})})).catch((function(e){console.log(e)}))},t}(o.a.Component);t.a=r},83:function(e,t,n){"use strict";var i,a=n(3),o=n(0),r=n.n(o),c=(n(77),n(344)),l=n(343),s=function(e){function t(t){var n;return(n=e.call(this,t)||this).state={codeString:""},n._currentFile=null,n}Object(a.a)(t,e),t.getDerivedStateFromProps=function(e,t){return e.id!==t.prevFile?{codeString:"",prevFile:e.file}:null};var n=t.prototype;return n.componentDidMount=function(){this._loadAsyncData(this.props.file)},n.componentDidUpdate=function(e,t){this.state.codeString||this._loadAsyncData(this.props.file)},n.componentWillUnmount=function(){this._currentFile=null},n._highlightLine=function(e){var t={display:"block"};return this.props.lines&&this.props.lines.includes(e)&&(t.backgroundColor="rgb(144, 202, 249, 0.15)"),{style:t}},n.render=function(){return r.a.createElement("div",{class:"code-block"},r.a.createElement(c.a,{language:"python",lineProps:this._highlightLine.bind(this),wrapLines:!0,lineNumberStyle:{color:"#80d6ff"},style:l.a,showLineNumbers:!0,customStyle:d,codeTagProps:{style:{color:"#e0e0e0"}}},this.state.codeString))},n._loadAsyncData=function(e){var t=this;this._currentFile=e,fetch("/pytorch-for-information-extraction/code-snippets/"+e+".py").then((function(e){return e.text()})).then((function(n){e===t._currentFile&&t.setState({codeString:n})})).catch((function(e){console.log(e)}))},t}(r.a.Component);t.a=s;var d=((i={borderRadius:0,overflow:"auto",maxHeight:"75vh",fontSize:"0.67em"}).borderRadius=8,i)}}]);