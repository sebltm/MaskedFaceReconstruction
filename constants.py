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
# imdir = "logs/train_data/" + datetime.now().strftime("%Y%m%d-%H%M%S")
imdir = "logs/train_data/20210817-101948"
histdir = "logs/hist/" + type_str + "-mid-" + str(mid_caps) + "-" + str(dims) + \
          "-out-" + str(output_caps) + "-" + str(dims)

restore_disc_triplet = True
restore_disc_began = True
restore_gen = True
