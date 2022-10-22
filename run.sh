# cifar 83% test acc
python main.py --no-wandb --epochs 10 --batch-size 128 --learning-rate 0.01 --model AVModel

# cifar 60 %
python main.py --no-wandb --epochs 10 --batch-size 128 --learning-rate 0.01 --model SimpleModel
