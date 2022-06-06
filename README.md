# Vision3d-Engine: An Easy-to-use yet Powerful Training Engine from Vision3d

`vision3d-engine` is an easy-to-use yet powerful training engine for fast prototyping and experiments. The main features include:

1. Simple but generic interface for training, validation, testing and debugging.
2. Two modes available for training: epoch-based (lr is adjusted every epoch) and iteration-based (lr is adjusted every iteration).
3. `dict` is used for message passing between different modules during training: dataloaders, models, loss functions, etc.
4. Pseudo batching (gradient accumulation).
5. Automatic `DistributedDataParallel` for Mult-GPU training.
6. Automatic checkpointing and model selection.
7. Automatic logging and TensorBoard support.
8. Intermediate endpoint tensors management.

Note: This repo will be merged into [Vision3d](https://github.com/qinzheng93/vision3d) in the future.

## Installation

Use the following command for installation:

```bash
python setup.py develop
```

## Trainer

There are two basic trainer out-of-box: `EpochBasedTrainer` and `IterBasedTrainer`. To train a model, a custom trainer class should be inherited from one of the basic trainer. For both trainers, you need to:

1. Call `register_model` to register a model.
2. Call `register_optimizer` to register a optimizer.
3. Optionally call `register_scheduler` to register a lr scheduler. 
4. Call `register_loader` to register a training and a validation dataloer.
5. Implment `train_step` and `val_step`.

### 1. EpochBasedTrainer

The training pipeline of `EpochBasedTrainer` is:

```text
1. before_train_epoch
2. for each iteration:
    2.1 before_train_step
    2.2 train_step
    2.3 after_backward
    2.4 optimizer_step
    2.5 after_train_step
    2.6 log iteration
3. scheduler_step
4. after_train_epoch
```

### 2. IterBasedTrainer

The training pipeline of `IterBasedTrainer` is:

```text
1. before_train_epoch
2. for each iteration:
    2.1 before_train_step
    2.2 train_step
    2.3 after_backward
    2.4 optimizer_step
    2.5 after_train_step
    2.6 log iteration
    2.7 scheduler_step
3. after_train_epoch
```

### 3. API

Training API:

* `before_train_epoch(self, epoch)`: Function before the training epoch.
* `before_train_step(self, epoch, iteration, data_dict)`: Function before the training step.
* `train_step(self, epoch, iteration, data_dict)`: Training step.
  * Must be implemented.
  * Return a model output dict and a result dict.
  * The result dict must contain a "loss" keyword of the overall loss.
  * The items in the result dict will be recorded for logging, so make sure all of them contain only one value.
* `after_backward(self, epoch, iteration, data_dict, output_dict, result_dict)`: After backward function for gradient debugging.
* `after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict)`: Function after the training step.
* `after_train_epoch(self, epoch, summary_dict)`: Function after the training epoch.
  * `summary_dict` contains the mean values of the result dict over the entire epoch.

Validation API:

* `before_val_epoch(self, epoch)`: Function before the validation epoch.
* `before_val_step(self, epoch, iteration, data_dict)`: Function before the validation step.
* `val_step(self, epoch, iteration, data_dict)`: Validation step.
  * Must be implemented.
  * Return a model output dict and a result dict.
  * The items in the result dict will be recorded for logging, so make sure all of them contain only one value.
* `after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict)`: Function after the validation step.
* `after_val_epoch(self, epoch, summary_dict)`: Function after the validation epoch.
  * `summary_dict` contains the mean values of the result dict over the entire epoch.

## Tester

There are two basic testers: `SingleTester`, `BatchTester`. `SingleTester` is for the cases where the batch size is 1. And `BatchTester` is for the cases where the batch size is larger than 1. Similarly, a custom tester must be inherited from one of the basic testers. For both testers, you need to:

1. Call `register_model` to register a model.
2. Call `register_loader` to register a testing dataloer.
3. Implment `test_step` and `eval_step`.

### 1. SingleTester

* `before_test_epoch(self)`: Function before the testing epoch.
* `before_test_step(self, iteration, data_dict)`: Function before the testing step.
* `test_step(self, iteration, data_dict)`: Testing step.
  * Must be implemented.
  * Return a dict of the model output.
* `eval_step(self, iteration, data_dict, output_dict)`: Evaluation step.
  * Must be implemented.
  * Return a dict of the model evaluation.
  * The items in the result dict will be recorded for logging, so make sure all of them contain only one value.
* `after_test_step(self, iteration, data_dict, output_dict, result_dict)`: Function after the testing step.
* `after_test_epoch(self, summary_dict)`: Function after the testing epoch.
  * `summary_dict` contains the mean values of the result dict over the entire epoch.

### 2. BatchTester

* `before_test_epoch(self)`: Function before the testing epoch.
* `before_test_step(self, iteration, data_dict)`: Function before the testing step.
* `test_step(self, iteration, data_dict)`: Testing step.
  * Must be implemented.
  * Return a dict of the model output for the testing **batch**.
* `split_batch_dict(self, iteraion, data_dict, output_dict)`: Function split a batch into a list of single cases.
  * Return a list of data dicts and a list of output dicts for the single cases.
* `eval_step(self, iteration, data_dict, output_dict)`: Evaluation step for **a single case**.
  * Must be implemented.
  * Return a dict of the model evaluation.
  * The items in the result dict will be recorded for logging, so make sure all of them contain only one value.
* `after_test_step(self, iteration, data_dict, output_dict, result_dict)`: Function after the testing step for **a single case**.
* `after_test_epoch(self, summary_dict)`: Function after the testing epoch.
  * `summary_dict` contains the mean values of the result dict over the entire epoch.

## Utilities

### 1. ContextManager

### 2. ArgumentParser

### 3. Profiling

