# cifar 89% test acc
python main.py --epochs 100 --batch-size 32 --learning-rate 0.001 --model AVModel --dataset cifar_10

#
python main.py --epochs 100 --batch-size 32 --learning-rate 0.001 --model AVModel --dataset cifar_10 --save-as longexp

#
python main.py --epochs 100 --batch-size 32 --learning-rate 0.001 --model AVModel --dataset cifar_10 --save-as longexp2 --resume-from-saved longexp

# cifar 60%
python main.py --epochs 10 --batch-size 128 --learning-rate 0.01 --model SimpleModel --dataset cifar_10

# cifar 81% densenet
python main.py --epochs 100 --batch-size 32 --learning-rate 0.01 --model DenseNet --dataset cifar_10

# mnist 98%
python main.py --epochs 10 --batch-size 128 --learning-rate 0.01 --model SimpleMNIST --dataset mnist

# 
python main.py --epochs 100 --batch-size 128 --learning-rate 0.01 --model SimpleDLA --dataset cifar_10

#
python main.py --epochs 100 --batch-size 128 --learning-rate 0.01 --model SimpleDLA --dataset cifar_10 --save-as longexp

#
python main.py --epochs 100 --batch-size 16 --learning-rate 0.001 --model SimpleDLA --dataset cifar_10 --save-as longexp2 --resume-from-saved longexp