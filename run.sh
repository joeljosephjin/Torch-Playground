# cifar 83% test acc
python main.py --no-wandb --epochs 10 --batch-size 128 --learning-rate 0.01 --model AVModel --dataset cifar_10

# cifar 60%
python main.py --no-wandb --epochs 10 --batch-size 128 --learning-rate 0.01 --model SimpleModel --dataset cifar_10

# cifar 81% densenet
python main.py --no-wandb --epochs 10 --batch-size 32 --learning-rate 0.01 --model DenseNet --dataset cifar_10

# mnist 98%
python main.py --no-wandb --epochs 10 --batch-size 128 --learning-rate 0.01 --model SimpleMNIST --dataset mnist
