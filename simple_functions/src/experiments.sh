# for seed in 1 2 3
#   do
#     for setting in skewed_linear_regression linear_regression decision_tree sparse_linear_regression relu_2nn_regression
#         do
#           for model in s4 gpt2 mamba
#             do
#               # Job to perform
#               PYTHONPATH=../../. python train.py --config conf/${setting}_${model}.yaml --training.seed ${seed}
#             done
#       done
# done

# # Print some Information about the end-time to STDOUT
# echo "DONE";
# echo "Finished at $(date)";

#!/bin/bash

for seed in 1 2 3
do
  for setting in linear_regression gaussian_kernel_regression nonlinear_dynamical_system
  do
    for model in gpt2 mamba
    do
      PYTHONPATH=../../. python train.py --config conf/${setting}_${model}.yaml --training.seed ${seed}
    done
  done
done

echo "DONE"
echo "Finished at $(date)"

# chmod +x run_icl_experiments.sh && ./run_icl_experiments.sh