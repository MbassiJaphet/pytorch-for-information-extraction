# select hardware use for computations
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

load_detection_checkpoint = True
save_detection_checkpoint = True

detection_checkpoint_path = os.path.join('checkpoints', 'detection_mask_rcnn_resnet50.pth.tar')

'''
Do not edit the lines below
'''

if load_detection_checkpoint :
  detection_checkpoint = torch.load(detection_checkpoint_path) if os.path.exists(detection_checkpoint_path) else None
  detection_model_state_dict = detection_checkpoint['model_state_dict'] if not detection_checkpoint == None else None
  detection_optimizer_state_dict = detection_checkpoint['optimizer_state_dict'] if not detection_checkpoint == None else None
else :
  detection_checkpoint, detection_model_state_dict, detection_optimizer_state_dict = None, None, None

if not save_detection_checkpoint : detection_checkpoint_path = None
# initialize detection model using the state dictionary from checkpoint
detection_model = get_instance_segmentation_model(num_detection_classes, state_dict = detection_model_state_dict).to(device)

if detection_checkpoint == None : print('No checkpoint loaded ! Loaded pre-trained model instead...')
else : 
  print('Loaded model from checkpoint...')
  utils.checkpoint_summary(detection_checkpoint)