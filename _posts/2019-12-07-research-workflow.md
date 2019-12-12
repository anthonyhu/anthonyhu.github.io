---
layout: post
title: Research workflow
permalink: research-workflow
comments: true
github: https://github.com/anthonyhu/ml-research
---
<p align='center'><img src='/img/research-workflow.jpg' alt='research-workflow'/></p>

This guide contains good practices for a machine learning researcher to structure day-to-day work. It's broken down 
into four sections, that can each be read independently depending on your needs.  

### Content
1. [Code structure]({% post_url 2019-12-07-research-workflow%})
2. [Run experiments]({% post_url 2019-12-06-run-experiments%})
3. [Python environment]({% post_url 2019-12-07-python-environment%})
4. [Remote work]({% post_url 2019-12-07-remote-work%})

The most important section is __1. Code structure__ as it covers how to design machine learning code, with a detailed 
example. Anyone who want to get started, or better organise their codebase, should find this section useful.

The __remaining sections__ (2. Run experiments, 3. Python environment, 4. Remote work) should benefit those
who work day-to-day on machine learning research (e.g. run and organise experiments on multiple machines, while working
remotely).

-----

## 1. Code structure
Quality code is essential for good research. We should aim at building a codebase that becomes increasingly richer while 
remaining easy to use. We're going to discuss the structure of a machine learning codebase in 
_PyTorch_, but it's also
applicable to other frameworks (_TensorFlow, Keras, Caffe_..).

In machine learning, we first prepare the __data__, then define the __model architecture + loss__, and 
finally __train__ the model. This is where we should focus our research efforts on.

_Everything else_ (monitoring metrics on tensorboard, saving/restoring model weights, printing log outputs etc.) should 
only be implemented once, and reused across projects. 

We're going to implement a general `Trainer` class that contains all the training logic (i.e.
_everything else_). Whenever we want to start a new machine learning project, we simply need to inherit from `Trainer` 
and implement the data and model creation. To illustrate how straightforward it is, we'll go through a detailed example 
shortly.
 
### 1.1. Trainer initialisation
First, let's go step by step in the init function of the general `Trainer` class. The only argument 
of the trainer is the path to a __config file__ (more details on the config file in 
[Section 2]({% post_url 2019-12-06-run-experiments%})), that contains all the training settings (batch size, 
number of workers, learning rate etc.) and the hyperparameters of the model.

```python
class Trainer:
    def __init__(self, config):
        self.config = config
        self.session_name = None
        # Initialise the training session by creating a new folder named `self.session_name`
        self.initialise_session()
        
        # Monitor training with tensorboard
        self.tensorboard = SummaryWriter(self.session_name)
        # Use the gpu if available
        self.device = torch.device('cuda') if self.config.gpu else torch.device('cpu')
```

A new folder will be created each time an experiment is ran. The name of this folder follows the format
 `session_{machine_name}_{time}_{tag}`, with the tag (contained in the config file) 
specifying the name of the experiment (e.g. `baseline`). 

This folder will contain: a copy of the config file used 
to create the training session (to easily reproduce experiments), checkpoints of the model/optimiser (to restore weights), a tensorboard 
file (to monitor metrics) and a .txt log file saving all the terminal outputs.

```python
        # Data
        self.train_dataset, self.val_dataset = None, None
        self.train_dataloader, self.val_dataloader = None, None
        # Abstract method that initialises the PyTorch Dataset and DataLoader classes 
        self.create_data()
        
        # Model
        self.model = None
        # Build the neural network and move it to the desired device
        self.create_model()
        self.model.to(self.device)
        
        # Loss
        self.loss_fn = None
        # Instantiate the loss function
        self.create_loss()
        
        # Optimiser
        self.optimiser = None
        # Initialise the optimiser
        self.create_optimiser()
        
        # Metrics
        self.train_metrics = None
        self.val_metrics = None
        # What we monitor during training on both the train and validation sets
        self.create_metrics()
```

For each new project, we simply need to implement the abstract methods `self.create_data`, `self.create_model`, 
`self.create_loss`, `self.create_optimiser` and `self.create_metrics`. The general `Trainer` class will handle 
everything else. We will shortly show (in [subsection 1.4](#14-example)) an example implementation on [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html), 
a classification dataset of tiny 32x32 images.

### 1.2. Train step
Now let's see how the trainer computes one training step. This is the method `Trainer.train_step`:
```python
def train_step(self):
    # Fetch a training batch. `batch` is a dictionary containing all the inputs and labels.
    batch = self._get_next_batch()
    # Cast the batch to the correct device
    self.cast_to_device(batch)
    
    # Forward pass
    output = self.forward_model(batch)
    loss = self.forward_loss(batch, output)
    
    # Backward pass
    self.optimiser.zero_grad()
    loss.backward()
    self.optimiser.step()
    
    # Print a log output to the terminal, and save loss on tensorboard.
    self.print_log(loss)
    self.tensorboard.add_scalar('train/loss', loss.item(), self.global_step)

    # Visualisation
    self.visualise(batch, output, 'train')
```

The log output printed in the terminal looks like:
```
Iteration  100/10000 | examples/s: 7785.4 | loss: 1.3832 | time elapsed: 00h00m02s 
                     | time left: 00h04m46s
Fetch data time: 2ms, model update time: 7ms
```

We monitor how long fetching one batch takes (2ms) -- if it's too slow, we might need more workers -- and how long
a single model update takes (7ms). Optimising these two values will result in an overall lower training time (indicated
by 'time left'). 

The `train_step` method is very general and operates with any input `batch`: a python dictionary, 
created by the data loader `self.train_dataloader`, containing the inputs and labels of the model.

### 1.3 Training the model
The main method of the trainer is `Trainer.train`: it optimises and evaluates the model, outputs the metrics and 
visualisation, and saves checkpoints regularly.

```python
def train(self):
    while self.global_step < self.config.n_iterations:
        self.global_step += 1
        self.train_step()

        if self.global_step % self.config.val_iterations == 0:
            # Evaluate the model on the validation set
            score = self.test()

            if score > self.best_score:
                self.best_score = score
                self.save_checkpoint()
```

### 1.4. Example
In practice, simply fork my [repository]({{ page.github }}) 
and implement the abstract methods of the trainer. For illustration, the repository contains an example that trains a CIFAR10 
model with only a few lines of code:

```python
# Inherit from the general `Trainer` class
class CifarTrainer(Trainer):
    # Implement all the abstract classes.
    def create_data(self):
        # Load dataset containing input 32x32 images and corresponding labels.
        self.train_dataset = CifarDataset(mode='train')
        self.val_dataset = CifarDataset(mode='val')
        
        # Create batches using DataLoader
        self.train_dataloader = DataLoader(self.train_dataset, self.config.batch_size, 
                                           num_workers=self.config.n_workers, shuffle=True)
        self.val_dataloader = DataLoader(self.val_dataset, self.config.batch_size, 
                                         num_workers=self.config.n_workers, shuffle=False)

    def create_model(self):
        # A simple convolutional net.
        self.model = CifarModel()

    def create_loss(self):
        self.loss_fn = nn.CrossEntropyLoss()

    def create_optimiser(self):
        # Parameters of the model that are optimisable.
        parameters_with_grad = \
            filter(lambda p: p.requires_grad, self.model.parameters())
        # Use an Adam optimiser with L2 regularisation.
        self.optimiser = Adam(parameters_with_grad, self.config.learning_rate, 
                              weight_decay=self.config.weight_decay)

    def create_metrics(self):
        # Monitor the accuracy of our model (percentage of correctly classified images).
        self.train_metrics = AccuracyMetrics()
        self.val_metrics = AccuracyMetrics()

    def forward_model(self, batch):
        return self.model(batch['image'])

    def forward_loss(self, batch, output):
        return self.loss_fn(output, batch['label'])

    def visualise(self, batch, output, mode):
        # Visualise the input images to our model.
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
Iteration  300/10000 | examples/s: 5816.1 | loss: 1.0277 | time elapsed: 00h00m09s 
                     | time left: 00h03m39s
Fetch data time: 2ms, model update time: 11ms

...
```

If the training is interrupted, it can be resumed by pointing to the path of the experiment (the folder 
whose name is `self.session_name` that was created in the init function of the `Trainer`). Running
`python run_training.py --restore /path/to/experiment/` will restore the weights of the model and optimiser, and
continue training where we left it.

Next we will cover how to run [reproducible experiments]({% post_url 2019-12-06-run-experiments%}), 
how to setup a reliable [python environment]({% post_url 2019-12-07-python-environment%}), and how to productively
[work remotely]({% post_url 2019-12-07-remote-work%}).

-----
_Big thanks to the Wayve team, who taught me how to effectively structure my code._
