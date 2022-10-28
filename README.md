# general library for experimenting Computer Vision

Prior libraries for image classification, hyperparameter tuning, report creation, etc..]

### formats

Dataloader: should be of same specific format as loading inbuilt mnist or cifar

### fundamental questions?

1.Does hyperparameter tuning on small portion of the dataset generalize to the big dataset?

2.What optimizers are useful for which datasets? Does one have more stability than other?

3.Benefits of fine-tuning on pre-trained models

4.Reproduce results of research papers and do ablation studies on them.

5.try on new research datasets

### prior work

Prior libraries don’t do work on this simple level. Our library will especially help beginners dive into computer vision model building, training and research

### code structure

^baseline
- main.py (train and test loop, hyperparameters, etc..)
- 

^future:
- train.py (will contain training loop function)
- test.py (will contain testing loop function)

### future work

- remove config.py (done)
- add option to select model (done)
- add mnist as a dataset option (done)
- use config for selecting the model (done)
- add more datasets? (later)
- reproduce a research paper results (in progress)
- code densenet from scratch from the paper and tensorflow implementations if needed (in progress)
- add readymade densenet to pipeline (done)
- investigate what makes densenet better and how i can find this phenomenon with less summative running time (to do)
- can another regular model with comparable number of parameters give same effect? How much is the effect of their special system (to do)
- remove usage pipeline.py file (in progress)
