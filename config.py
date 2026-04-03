import os

IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 350
LR = 5e-5
WEIGHT_DECAY = 1e-4
SEED = 4

LAST_K = 2


DATA_DIR = ''   

OUTPUT_DIR = './outputs'
CKPT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')

NUM_CLASSES = 2
EMBED_DIM = 256
