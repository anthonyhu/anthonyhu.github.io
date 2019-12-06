---
layout: post
title: PhD research workflow
permalink: phd-workflow
comments: true
---
I just finished the first year of my PhD and I've decided to write a guide on good practices for a PhD researcher to 
structure day-to-day work. This guide is broken down into four sections, that can each be read independently depending 
on your needs. 

### Contents
1. Code structure
2. Run experiments
3. Python environment
4. Remote workflow

-----

## 1. Code structure
Quality code is essential for good research. Throughout our PhD, we should aim at building a codebase that becomes 
increasingly richer while remaining easy to use. I'll discuss the structure of a machine learning codebase in 
PyTorch, but it may be applicable to other fields.

In machine learning, we first prepare the __data__, then define the __model architecture and loss function__, and 
finally we __train the model__ with an optimiser. This is what we should focus our research efforts on.

All the rest (monitoring metrics on tensorboard, saving/restoring model weights, printing log outputs, 
checking how much time data preprocessing/
forward/backward pass takes, how much longer the model needs to train etc.) should only be implemented once, and reused
across models. This blog post presents a code structure where the mentioned helping tools are implemented, enabling us
to only spend time on actual research.

We are going to use a general `Trainer` class that implements all the training logic. The only argument 
to the trainer is the path to a config file, that contains all the training settings (batch size, number of workers, 
learning rate etc) and the hyperparameters of the model to easily reproduce experiments and restore models.

I will now briefly outline the code structure. The full implementation is available on the following [github 
repository](https://github.com/anthonyhu/vision). 

### 1.1. Trainer initialisation
Let us go step by step in the init function of the general `Trainer` class.

#### Session
```python
self.config = None
self.session_name = ''
# Initialise the session by creating a new folder named self.session_name for 
# the experiment specified by the config file
self.initialise_session()

self.tensorboard = SummaryWriter(self.session_name, comment=self.config.tag)
self.device = torch.device('cuda') if self.config.gpu else torch.device('cpu')
```

The name of the folder follows the format `session_{machine}_{time}_{tag}`, with the tag specifying the name of 
that particular experiment. 
For example `session_direwolf_2019_12_05_16_40_34_baseline/`. In this folder, we will have the config file used 
to create the training session, checkpoints of the model/optimiser, tensorboard 
file and a log file that contains all the terminal outputs during training.

#### Data
```python
# Initialise the pytorch Dataset and DataLoader classes
self.train_dataset, self.val_dataset = None, None
self.train_dataloader, self.val_dataloader = None, None
self.create_data()
```

#### Model
```python
# Build the neural network and move it to the desired device
self.model = None
self.create_model()
print_model_spec(self.model)
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

The abstract methods `self.create_data`, `self.create_model()`, etc., need to be implemented for each new project, and
they are the only thing that needs to be implemented. The `Trainer` class will handle everything else. We will show
shortly an example implementation on CIFAR.

### 1.2. Train step
Now let us see how the model computes one training step. This is the function `Trainer.train_step()`.
```python
# Fetch a training batch
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
    step_duration = time() - t0
    self.print_log(loss, step_duration, data_fetch_time, model_update_time)
    self.tensorboard.add_scalar('train/loss', loss.item(), self.global_step)

# Visualise
if self.global_step % self.config.vis_iterations == 0:
    self.train_metrics.update(output, batch['label'])
    self.visualise(batch, output, 'train')
```

The log output looks like:
```
Iteration  100/10000 | examples/s: 7785.4 | loss: 1.3832 | time elapsed: 00h00m02s 
                     | time left: 00h04m46s
Fetch data time: 2ms, model update time: 7ms
```

The `train_step` function is very general and work with any input `batch`, which is a Python dictionary containing the
inputs of the model as well as the labels (for supervised training) given by the PyTorch DataLoader. 

### 1.3. How to train a new model
In practice, simply fork my [repository](https://github.com/anthonyhu/vision) 
and implement the following abstract methods of the trainer. The repository contains an example that trains a CIFAR10 
model in only a few lines of code:

```python
class CifarTrainer(Trainer):
    def create_data(self):
        self.train_dataset = CifarDataset(DATA_ROOT, mode='train')
        self.val_dataset = CifarDataset(DATA_ROOT, mode='val')

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
Fetch data time: 10ms, model update time: 7ms

Iteration  200/10000 | examples/s: 7326.7 | loss: 1.3379 | time elapsed: 00h00m04s 
                     | time left: 00h03m58s
Fetch data time: 8ms, model update time: 9ms

Iteration  300/10000 | examples/s: 7199.7 | loss: 1.1732 | time elapsed: 00h00m06s 
                     | time left: 00h03m42s
Fetch data time: 8ms, model update time: 10ms

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
Fetch data time: 8ms, model update time: 11ms

Iteration  500/10000 | examples/s: 5675.3 | loss: 1.0342 | time elapsed: 00h00m11s 
                     | time left: 00h03m32s
Fetch data time: 10ms, model update time: 8ms
...
```

Next we will cover how to run and organise [reproducible experiments]({% post_url 2019-12-06-run-experiments%}), 
how to setup a reliable [python environment]({% post_url 2019-12-06-python-environment%}), and how to productively
[work remotely]({% post_url 2019-12-06-remote-workflow%}).


-----
_Big thanks to Wayve, who taught me how to effectively structure my code._ 



