# select hardware type use for computations
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

load_orientation_checkpoint = True
save_orientation_checkpoint = True

orientation_checkpoint_path = os.path.join('checkpoints', 'orientation_model_scratch.pth.tar')

'''
Do not edit the lines below
'''

if load_orientation_checkpoint:
  orientation_checkpoint = torch.load(orientation_checkpoint_path) if os.path.exists(orientation_checkpoint_path) else None
  orientation_model_state_dict = orientation_checkpoint['model_state_dict'] if not orientation_checkpoint == None else None
  orientation_optimizer_state_dict = orientation_checkpoint['optimizer_state_dict'] if not orientation_checkpoint == None else None
else:
  orientation_checkpoint, orientation_model_state_dict, orientation_optimizer_state_dict = None, None, None

if not save_orientation_checkpoint : orientation_checkpoint_path = None
# initialize orientation model using the state dictionary from checkpoint
orientation_model = get_orientation_model(num_orientation_classes, model_state_dict = orientation_model_state_dict).to(device)

if orientation_checkpoint is None : print('No checkpoint loaded ! Initialized model from scratch instead...')
else : 
  print('Loaded model from checkpoint...')
  utils.checkpoint_summary(orientation_checkpoint)