LIMIT_IMAGES = 69445 #number of image to download, 69445 is the max
DELTA_PREPROCESSING = 25

LIMIT_IMAGES_SEGMENTATION_DOWNLOAD = 5000
LIMIT_IMAGES_SEGMENTATION_PKL = 3500
TEST_SET_SIZE_SEGMENTATION = 200
IMG_CHANNELS_UNET = 1
IMG_SIZE_UNET = 256
EPOCHS_UNET = 30
LR_UNET = 0.0005
BS_UNET = 32
FUNCTION_UNET = 'relu'

#LIMIT_IMAGES_CLASSIFICATION_PKL = 10
#TEST_SET_SIZE_CLASSIFICATION = 2
LIST_DATASET_NAME = ["UDA-1", "UDA-2", "MSK-1", "MSK-2", "MSK-3", "MSK-4", "2018 JID Editorial Images", "HAM10000", "ISIC_2020_Vienna_part_1", "BCN_20000", "ISIC_2020_Vienna_part2", "ISIC 2020 Challenge - MSKCC contribution", "Sydney (MIA / SMDC) 2020 ISIC challenge contribution", "BCN_2020_Challenge"]
IMG_CHANNELS_VGG = 3
IMG_SIZE_VGG = 224
EPOCHS_VGG = 50
LR_VGG = 0.0005
BS_VGG = 20
FUNCTION_VGG = 'sigmoid'
SIZE_TEST_SET = 100
NUMBER_IMG_FOR_CLASS = 5000

IMG_SIZE_RESNET = 224
EPOCHS_RESNET = 50
LR_RESNET = 0.0001
BS_RESNET = 20
FUNCTION_RESNET = 'sigmoid'

IMG_SIZE_BEMBONET = 224
EPOCHS_BEMBONET = 50
LR_BEMBONET = 0.001
BS_BEMBONET = 20
FUNCTION_BEMBONET = 'sigmoid'