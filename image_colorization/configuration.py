from image_colorization.nets.fcn_models import FCN_net1


dataset_path = 'datasets/Cifar-10/cifar-10-batches-py'

which_version = "V23"
which_epoch_version = 0

load_net_file = f"model_states/fcn_model{which_version}_epoch{which_epoch_version}.pth"
load_optimizer_file = f"model_states/fcn_optimizer{which_version}_epoch{which_epoch_version}.pth"
load_scheduler_file = f"model_states/fcn_scheduler{which_version}_epoch{which_epoch_version}.pth"

log_file = f"logs/logs_fcn_model{which_version}_train.log"

init_epoch = 0
how_many_epochs = 10
do_load_model = False

batch_size = 128
learning_rate = 0.1
momentum = 0.9
lr_step_scheduler = 1
lr_step_gamma = 0.999
step_decay = 0.5
decay_after_steps = 20

do_blur_processing = True
choose_train_dataset = True
ab_chosen_normalization = "standardization"
ab_output_normalization = "trick"
L_chosen_normalization = "standardization"

chosen_net = FCN_net1()

gauss_kernel_size = (7, 7)

# plot_lab = True
do_save_results = True
do_show_results = True
how_many_results_to_generate = 30
