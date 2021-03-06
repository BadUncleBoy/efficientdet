from utils.utils import get_objects

dataset_name = "kdxf"
compound_coef = 0
train_set   = "train"
val_set     = "val"

anchor_free_mode = False #whether to use anchor free method or not
''' training setting '''
####################################################
num_gpus    = 4
batch_size = 12

lr         = 1e-3
optim      = "adamw"
num_epochs = 500
num_workers   = 0

val_interval = 1
start_interval = 10 
unresotre_weights_dict = ["header"]
head_only    = True

load_weights = True
pret_weight_path = "./pretrained_weights/efficientdet/efficientdet-d0.pth"
data_path    = "/data/zy"
log_path     = "./logs"
saved_path    = "./weights"

es_min_delta = 0.0
es_patience = 0
debug        = False


patience     = 5 #当连续n个epoch训练集的loss不下降，则降低学习率
factor       = 0.5 # new_lr = old_lr * factor
min_lr       = 1e-6
#################################################


mean        =[0.750033, 0.811906, 0.791236]
std         =[0.266911, 0.231149, 0.281710]
# mean        =[0.485, 0.456, 0.406]
# std         =[0.229, 0.224, 0.225]

anchors_scales = '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios = '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

obj_list = get_objects(dataset_name)


'''	test setting '''
##############################################
force_input_size = None
img_path = 'demo_jpg/img.png'
weight_path = 'weights/kdxf/efficientdet-d0_24_6000.pth'
threshold = 0.3
iou_threshold = 0.2
################################################


''' eval setting'''
####################################################
eval_weight_path = 'weights/kdxf/efficientdet-d0_153_38000.pth'
eval_nms_threshold   = 0.5
eval_use_cuda = True
eval_gpu = 0
eval_use_float16 = False
eval_threshold = 0.02
###################################################
