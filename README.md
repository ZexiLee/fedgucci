# FedGuCci
## Usage
### Requirements
Install all the packages from requirements:
```bash
pip install -r requirements.txt
```
### Run
Execute the following code to run FedGuCci on CIFAR-100 using ResNet20:
```bash
python main.py --device 0 --random_seed 11 --local_model ResNet20 --dataset cifar100 --T 200 --batchsize 64 --lr 0.03 --node_num 50 --dirichlet_alpha 0.5 --E 3 --client_method multi_step_group_connectivity --server_method fedavg --table CV --inner_table cifar100
```