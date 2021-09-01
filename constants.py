from datetime import datetime

IMAGE_SIZE = 256

### SETUP CAPS ###
caps = True
type_str = "caps" if caps else "vgg-16"

output_caps = 10
mid_caps = 8
dims = 10

### SETUP TRAINING VARIABLES ###
batch_size = 1
steps = 5000
epochs = 1000

### LOGGING ###
log = True
log_vars = False

logdir = "logs/scalars/" + type_str + "-mid-" + str(mid_caps) + "-" + str(dims) + \
         "-out-" + str(output_caps) + "-" + str(dims)
imdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
histdir = "logs/hist/" + type_str + "-mid-" + str(mid_caps) + "-" + str(dims) + \
          "-out-" + str(output_caps) + "-" + str(dims)

restore_disc_triplet = False
restore_disc_began = False
restore_gen = False
