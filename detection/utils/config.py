
cfg_blaze = {
    'name': 'Blaze',
    # origin anchor
    # 'min_sizes': [[16, 24], [32, 48, 64, 80, 96, 128]],
    'min_sizes': [[8, 11], [14, 19, 26, 38, 64, 149]], 
    # kmeans and evolving for 640x640
    'steps': [8, 16],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 1, # 'loc_weight': 1,    # color blaze: 1 2 3 2
    'cls_weight': 6, # 'cls_weight': 6,
    'landm_weight': 0.1, # 'landm_weight': 0.1, 
    'feat_weight': 100,
    'gpu_train': True,
    'batch_size': 16,
    'ngpu': 1,
    'epoch': 20,
    'decay1': 130,
    'decay2': 160,
    'decay3': 175,
    'decay4': 185,
    'image_size': 320,
    'num_classes': 22,
    'pretrain': True,
    'milestone': [10, 13, 16]
}



# cfg_blaze = {
#     'name': 'Blaze',
#     'variance': [0.1, 0.2],
#     'clip': False,
#     'loc_weight': 1, # 'loc_weight': 1,    # color blaze: 1 2 3 2
#     'cls_weight': 2, # 'cls_weight': 6,
#     'landm_weight': 3, # 'landm_weight': 0.1, 
#     'mask_weight': 2,
#     'gpu_train': True,
#     'batch_size': 128,
#     'ngpu': 1,
#     'epoch': 200,
#     'decay1': 130,
#     'decay2': 160,
#     'decay3': 175,
#     'decay4': 185,
#     'image_size': 320,
#     'num_classes': 2,
# }



