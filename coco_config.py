from utils.utils import get_objects

dataset_name = "cocods"
compound_coef = 0
train_set   = "train2017"
val_set     = "val2017"

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
save_interval = 1000
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


mean        =[0.485, 0.456, 0.406]
std         =[0.229, 0.224, 0.225]

anchors_scales = '[2 ** 0, 2 ** (1.0 / 3.0), 2 ** (2.0 / 3.0)]'
anchors_ratios = '[(1.0, 1.0), (1.4, 0.7), (0.7, 1.4)]'

obj_list = get_objects(dataset_name)


'''	test setting '''
##############################################
force_input_size = 1536
img_path = 'demo_jpg/img.png'
weight_path = '/data/zy/Efficient_pytorch/weights/cocods/efficientdet-d0_13_136000.pth'
threshold = 0.3
iou_threshold = 0.2
################################################


''' eval setting'''
####################################################
eval_weight_path = '/data/zy/Efficient_pytorch/weights/cocods/efficientdet-d0_16_160000.pth'
eval_nms_threshold   = 0.5
eval_use_cuda = True
eval_gpu = 0
eval_use_float16 = False
eval_threshold = 0.02
###################################################
