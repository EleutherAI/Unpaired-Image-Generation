# (the latent logvars are fixed to -4!!!)
python -m model.trainer --debug --config t2ivae_combined_cifar10_resnet152_diffusion_test_reconstruction_9_1
python -m model.trainer --debug --config t2ivae_combined_cifar10_diffusion_test_reconstruction_9_1_latent512
python -m model.trainer --debug --config t2ivae_combined_cifar10_diffusion_test_reconstruction_9_1_latent2048
python -m model.trainer --debug --config t2ivae_combined_cifar10_diffusion_test_reconstruction_9_1_lr5e-5
python -m model.trainer --debug --config t2ivae_combined_cifar10_diffusion_test_reconstruction_9_1