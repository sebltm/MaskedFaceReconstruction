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

log = True
log_vars = False
