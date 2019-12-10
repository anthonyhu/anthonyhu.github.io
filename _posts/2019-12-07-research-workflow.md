---
layout: post
title: Research workflow
permalink: research-workflow
comments: true
---

<p align='center'><img src='/img/research-workflow.jpg' alt='research-workflow'/></p>

This guide contains good practices for a machine learning researcher to structure day-to-day work. It's broken down 
into four sections, that can each be read independently depending on your needs.  

### Content
1. [Code structure]({% post_url 2019-12-07-research-workflow%})
2. [Run experiments]({% post_url 2019-12-06-run-experiments%})
3. [Python environment]({% post_url 2019-12-07-python-environment%})
4. [Remote work]({% post_url 2019-12-07-remote-work%})

The most important section is __1. Code structure__ as it covers how to design a machine learning model in 
PyTorch with a detailed example. Anyone who want to get started, or to better organise their codebase, should find this
section useful.

The __remaining sections__ (2. Run experiments, 3. Python environment, 4. Remote work) should benefit those
who work day-to-day on machine learning research (e.g. run and organise experiments on multiple machines, while working
remotely).

-----

## 1. Code structure
Quality code is essential for good research. We should aim at building a codebase that becomes increasingly richer while 
remaining easy to use. We'll discuss the structure of a machine learning codebase in PyTorch, but it may be applicable 
to other fields.

In machine learning, we first prepare the __data__, then define the __model architecture + loss__, and 
finally __train__ the model. This is where we should focus our research efforts on.

_Everything else_ (monitoring metrics on tensorboard, saving/restoring model weights, printing log outputs etc.) should 
only be implemented once, and reused across models. 

We're going to implement a general `Trainer` class that contains all the training logic (i.e. what we called 
_everything else_). Whenever we want to build a new model, we simply have to implement the data and model creation, and 
to illustrate how straightforward it is, we'll go through a detailed example. 
 
### 1.1. Trainer initialisation
Let us go step by step in the init function of the general `Trainer` class. The only argument 
to the trainer is the path to a config file, that contains all the training settings (batch size, number of workers, 
learning rate etc.) and the hyperparameters of the model to easily reproduce experiments.

#### Training session
```python
# Initialise the training session by creating a new folder named `self.session_name`
self.session_name = None
self.initialise_session()

self.tensorboard = SummaryWriter(self.session_name)
# Use the gpu if available
self.device = torch.device('cuda') if self.config.gpu else torch.device('cpu')
```

The name of the folder follows the format `session_{machine}_{time}_{tag}`, with the tag (contained in the config file) 
specifying the name of the experiment. This folder will contain: a copy of the config file used 
to create the training session, the checkpoints of the model/optimiser, the tensorboard 
file and a .txt log file with all the terminal outputs.

#### Data
```python
self.train_dataset, self.val_dataset = None, None
self.train_dataloader, self.val_dataloader = None, None
# Abstract method that initialises the PyTorch Dataset and DataLoader classes 
self.create_data()
```

#### Model
```python
# Build the neural network and move it to the desired device
self.model = None
self.create_model()
self.model.to(self.device)
```

#### Loss
```python
# Create the model loss function
self.loss_fn = None
self.create_loss()
```

#### Optimiser
```python
# Initialise the optimiser
self.optimiser = None
self.create_optimiser()
```

#### Metrics
```python
# Metrics we monitor during training on both the train and validation sets
self.train_metrics = None
self.val_metrics = None
self.create_metrics()
```

For each new project, we simply need to implement the abstract methods `self.create_data`, `self.create_model`, 
`self.create_loss`, `self.create_optimiser` and `self.create_metrics`. The general `Trainer` class will handle 
everything else. We will shortly show an example implementation on CIFAR10, a classification dataset of tiny 32x32 
images.

### 1.2. Train step
Now let us see how the model computes one training step. This is the function `Trainer.train_step()`.
```python
# Fetch a training batch. `batch` is a dictionary containing all the inputs 
# and labels.
batch = self._get_next_batch()
# Cast the batch to gpu
self.preprocess_batch(batch)

# Forward pass
output = self.forward_model(batch)
loss = self.forward_loss(batch, output)

# Backward pass
self.optimiser.zero_grad()
loss.backward()
self.optimiser.step()

# Print a log output to monitor training
if self.global_step % self.config.print_iterations == 0:
    self.print_log(loss)
    self.tensorboard.add_scalar('train/loss', loss.item(), self.global_step)

# Visualise
if self.global_step % self.config.vis_iterations == 0:
    self.visualise(batch, output, 'train')
```

The log output looks like:
```
Iteration  100/10000 | examples/s: 7785.4 | loss: 1.3832 | time elapsed: 00h00m02s 
                     | time left: 00h04m46s
Fetch data time: 2ms, model update time: 7ms
```

We monitor how long data preparation takes (if it's too slow, we might need more workers), here 2ms, and how much time
a single update takes, here 7ms. Optimising these two values will result in an overall lower training time (indicated
by 'time left'). 

The `train_step` function is very general and work with any input `batch`, which is a Python dictionary containing the
inputs of the model as well as the labels (for supervised learning) given by the PyTorch DataLoader. 

### 1.3 Training the model
The main method of the trainer is `Trainer.train`: it optimises the model, outputs the metrics and visualisation, 
and saves checkpoint during training.

```python
def train(self):
    # Set the model in train mode
    self.model.train()

    while self.global_step < self.config.n_iterations:
        self.global_step += 1
        self.train_step()

        if self.global_step % self.config.val_iterations == 0:
            score = self.test()

            if score > self.best_score:
                self.best_score = score
                self.save_checkpoint()
```

### 1.4. Example
In practice, simply fork my [repository](https://github.com/anthonyhu/vision) 
and implement the abstract methods of the trainer. The repository contains an example that trains a CIFAR10 
model with only a few lines of code:

```python
# Inherit from the general `Trainer` class
class CifarTrainer(Trainer):
    def create_data(self):
        self.train_dataset = CifarDataset(mode='train')
        self.val_dataset = CifarDataset(mode='val')

        self.train_dataloader = DataLoader(
            self.train_dataset, batch_size=self.config.batch_size, 
            num_workers=self.config.n_workers, shuffle=True)
        self.val_dataloader = DataLoader(
            self.val_dataset, batch_size=self.config.batch_size,
            num_workers=self.config.n_workers, shuffle=False)

    def create_model(self):
        self.model = CifarModel()

    def create_loss(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def create_optimiser(self):
        parameters_with_grad = \
            filter(lambda p: p.requires_grad, self.model.parameters())
        self.optimiser = Adam(parameters_with_grad, lr=self.config.learning_rate, 
                              weight_decay=self.config.weight_decay)

    def create_metrics(self):
        self.train_metrics = AccuracyMetrics()
        self.val_metrics = AccuracyMetrics()

    def forward_model(self, batch):
        return self.model(batch['image'])

    def forward_loss(self, batch, output):
        return self.loss_fn(output, batch['label'])

    def visualise(self, batch, output, mode):
        self.tensorboard.add_images(mode + '/image', batch['image'], 
                                    self.global_step)
```

Running `python run_training.py --config experiments/cifar.yml` then produces the following output:

```
Iteration  100/10000 | examples/s: 7785.4 | loss: 1.3832 | time elapsed: 00h00m02s 
                     | time left: 00h04m46s
Fetch data time: 2ms, model update time: 7ms

Iteration  200/10000 | examples/s: 7326.7 | loss: 1.3379 | time elapsed: 00h00m04s 
                     | time left: 00h03m58s
Fetch data time: 2ms, model update time: 9ms

Iteration  300/10000 | examples/s: 7199.7 | loss: 1.1732 | time elapsed: 00h00m06s 
                     | time left: 00h03m42s
Fetch data time: 3ms, model update time: 10ms

----------------------------------------------------------------------------------
Validation
----------------------------------------------------------------------------------
100%|█████████████████████████████████████████████| 79/79 [00:01<00:00, 47.14it/s]
Val loss: 1.1156
----------------------------------------------------------------------------------
Metrics
----------------------------------------------------------------------------------
Train score: 0.648
Val score: 0.596
New best score: -inf -> 0.596
Model saved to: /path/to/experiment/checkpoint

-----------------------------------------------------------------------------------
Iteration  400/10000 | examples/s: 5816.1 | loss: 1.0277 | time elapsed: 00h00m09s 
                     | time left: 00h03m39s
Fetch data time: 2ms, model update time: 11ms

Iteration  500/10000 | examples/s: 5675.3 | loss: 1.0342 | time elapsed: 00h00m11s 
                     | time left: 00h03m32s
Fetch data time: 3ms, model update time: 8ms
...
```

If the training is interrupted, it can be resumed by pointing to the path of the experiment:
`python run_training.py --restore /path/to/experiment/`.

Next we will cover how to run [reproducible experiments]({% post_url 2019-12-06-run-experiments%}), 
how to setup a reliable [python environment]({% post_url 2019-12-07-python-environment%}), and how to productively
[work remotely]({% post_url 2019-12-07-remote-work%}).


-----
_Big thanks to the Wayve team, who taught me how to effectively structure my code._ 
