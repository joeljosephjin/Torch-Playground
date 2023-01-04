# cifar 89% test acc
python main.py --epochs 100 --batch-size 32 --learning-rate 0.001 --model AVModel --dataset cifar_10

#
python main.py --epochs 100 --batch-size 32 --learning-rate 0.001 --model AVModel --dataset cifar_10 --save-as longexp

#
python main.py --epochs 100 --batch-size 32 --learning-rate 0.001 --model AVModel --dataset cifar_10 --save-as longexp2 --resume-from-saved longexp

#
python main.py --epochs 200 --batch-size 640 --learning-rate 0.1 --model AVModel --dataset cifar_10 --save-as longexp | tee save/AVModellongexp.txt

#
python main.py --epochs 200 --batch-size 128 --learning-rate 0.01 --model AVModel --dataset cifar_10 --save-as longexp --use-wandb

#
python main.py --epochs 200 --batch-size 512 --learning-rate 0.05 --model SimpleDLA --dataset cifar_10 --save-as longexp --use-wandb

# cifar 60%
python main.py --epochs 10 --batch-size 128 --learning-rate 0.01 --model SimpleModel --dataset cifar_10

# cifar 81% densenet
python main.py --epochs 100 --batch-size 32 --learning-rate 0.01 --model DenseNet --dataset cifar_10

# 
python main.py --epochs 200 --batch-size 256 --learning-rate 0.01 --model DenseNet --dataset cifar_10 --use-wandb

# mnist 98%
python main.py --epochs 10 --batch-size 128 --learning-rate 0.01 --model SimpleMNIST --dataset mnist

# 
python main.py --epochs 100 --batch-size 128 --learning-rate 0.01 --model SimpleDLA --dataset cifar_10

#
python main.py --epochs 100 --batch-size 128 --learning-rate 0.01 --model SimpleDLA --dataset cifar_10 --save-as longexp

#
python main.py --epochs 100 --batch-size 16 --learning-rate 0.001 --model SimpleDLA --dataset cifar_10 --save-as longexp2 --resume-from-saved longexp

# 93% - 7 hours
python main.py --epochs 200 --batch-size 128 --learning-rate 0.1 --model SimpleDLA --dataset cifar_10 --save-as longexp | tee save/SimpleDLAlongexp.txt

# siamese
python main-siamese.py --epochs 500 --batch-size 4096 --learning-rate 0.07 --weight-decay 5e-3

# few shot
python main-siamese-fewshot.py --epochs 500 --batch-size 4096 --learning-rate 0.001 --weight-decay 5e-3 --n-shot 1

# 74%
python main-siamese-fewshot.py --epochs 500 --batch-size 4096 --learning-rate 0.005 --weight-decay 1e-3
