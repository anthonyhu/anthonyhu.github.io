---
layout: post
title: PhD research workflow
comments: true
permalink: phd-workflow
---
I just finished the first year of my PhD and I've decided to write a guide with good practices for a PhD researcher to 
structure day-to-day work. This guide is broken down into four sections, that can each be read independently depending 
on your needs. 

### Contents

1. Code structure
2. Run experiments
3. Python environment
4. Remote workflow

In this post I will cover __Code structure__ and the remaining sections are available on separate blog posts.

-----

## Code structure

Good code is essential for good research. Throughout the PhD, we need to build a codebase that becomes increasingly 
richer while remaining easy to use. I'll outline the structure of a machine learning oriented codebase, but it may be 
applicable to other fields.

In machine learning, we first prepare the data, then we define the model and the loss function, and finally we train 
the model with an optimiser. These three steps are common to any machine learning, and thus shape the structure of our 
codebase.

I will now briefly outline the code structure, using PyTorch. More details are available on the following github 
repository: [https://github.com/anthonyhu/vision](https://github.com/anthonyhu/vision).

We are going to use a general `Trainer` class that implements all the necessary steps of training. The only argument 
to the trainer is the path to a config file, that contains all the training settings (batch size, number of workers, 
learning rate etc) and the hyperparameters of the model. 

### 1. Trainer initialisation
Let us go step by step in the init function.

#### Session
```python
self.config = None
self.session_name = ''
self.initialise_session()

self.tensorboard = SummaryWriter(self.session_name, comment=self.config.tag)
self.device = torch.device('cuda') if self.config.gpu else torch.device('cpu')
```

We initialise the session by creating a new folder for the experiment we ran. The name of the folder looks like 
`session_direwolf_2019_12_05_16_40_34_experiment_name/`. `direwolf` is the name of my machine. In this folder, 
we will have the config file used to create the training session, checkpoints of the model/optimiser and a tensorboard 
log.

#### Data
```python
self.train_dataset, self.val_dataset = None, None
self.train_dataloader, self.val_dataloader = None, None
# Initialise the pytorch Dataset and DataLoader classes
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
self.loss_fn = None
self.create_loss()
```

#### Optimiser
```python
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


### 2. Train step
Now let us see how the model computes one training step:
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

### 3. How to train a new model
In practice, you can fork my repository at [https://github.com/anthonyhu/vision](https://github.com/anthonyhu/vision) 
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

```python
def visualise(self, batch, output, mode)
```

This is `print("Hello world")`{.python} inline code.
