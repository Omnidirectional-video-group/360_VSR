"""
Module Description:

This script introduces learner


Author:

Ahmed Telili

Date Created:

29 01 2024

"""

import torch
import os
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from src.learner.metrics import WSPSNRMetric
from src.learner.saver import Saver
from src.learner.loss import *
import numpy as np
import time
import cv2

class StandardLearner():
    """
        config: A `dict` contains the configuration of the learner.
        model: A list of `pytorch` objects which generate predictions.
        dataset: A dataset `dict` contains dataloader for different split.
        step: An `int` represents the current step. Initialize to 0.
        optimizer: A `tf.keras.optimizers` is used to optimize the model. Initialize to None.
        lr_scheduler: A `tf.keras.optimizers.schedules.LearningRateSchedule` is used to schedule
            the leaning rate. Initialize to None.
        metric_functions: A `dict` contains one or multiple functions which are used to
            metric the results. Initialize to {}.
        saver: A `Saver` is used to save checkpoints. Initialize to None.
        summary: A `TensorboardSummary` is used to save eventfiles.
        log_dir: A `str` represents the directory which records experiments.
        steps: An `int` represents the number of train steps.
        log_train_info_steps: An `int` represents frequency of logging training information.
        keep_ckpt_steps: An `int` represents frequency of saving checkpoint.
        valid_steps: An `int` represents frequency  of validation.
    """

    def __init__(self, config, model, dataset, log_dir):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.log_dir = log_dir
        self.loss_fn = None
        self.step = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.metric_functions = {}
        self.saver = None
        self.summary = SummaryWriter(log_dir)

        # Initialize training components
        self.register_training()

        # Check if a checkpoint path is provided in the config and load it
        restore_path = config['learner']['saver']['restore']
        if restore_path is not None and restore_path != 'null':
            restore_full_path = Path(log_dir) / restore_path
            self.saver = Saver(self.model, self.optimizer, save_dir=Path(log_dir) / 'checkpoints')
            self.saver.load_checkpoint(restore_full_path)
            self.step = self.saver.step

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        # Set random seed for reproducibility
        torch.manual_seed(2454)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(2454)
            torch.cuda.manual_seed_all(2454)
        np.random.seed(2454)

    def register_training(self):
        # Extract optimizer configurations
        opt_config = self.config['learner']['optimizer']
        OptimizerClass = getattr(torch.optim, opt_config['name'])

        # Adjust the configuration
        betas = (opt_config.get('beta_1', 0.9), opt_config.get('beta_2', 0.999))
        opt_config_adjusted = {'lr': opt_config.get('lr', 0.001), 'betas': betas}

        # Initialize the optimizer with the adjusted configuration
        self.optimizer = OptimizerClass(self.model.parameters(), **opt_config_adjusted)

        # Extract lr_scheduler configurations
        lr_scheduler_config = self.config['learner']['lr_scheduler']
        if lr_scheduler_config['name'] == 'ExponentialDecay':
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                self.optimizer,
                gamma=lr_scheduler_config['decay_rate']
            )
        # Other lr_schedulers can be added here

        # Define loss function
        loss_config = self.config.get('learner', {}).get('loss', {})
        loss_name = loss_config.get('name', 'MSELoss').lower()

        if loss_name == 'charbonnierloss':
            self.loss_fn = CharbonnierLoss(**loss_config.get('params', {}))
        elif loss_name == 'perceptualloss':
            self.loss_fn = PerceptualLoss(**loss_config.get('params', {}))
        elif loss_name == 'ganloss':
            self.loss_fn = GANLoss(**loss_config.get('params', {}))
        elif loss_name == 'tvloss':
            self.loss_fn = TVLoss(**loss_config.get('params', {}))
        else:
            self.loss_fn = torch.nn.MSELoss()  # Default loss
        # Define metrics
        self.metric_functions['ws-psnr'] = WSPSNRMetric()

        if self.saver is None:  # Only initialize if not already set up by checkpoint loading
            self.saver = Saver(self.model, self.optimizer, save_dir=Path(self.log_dir) / 'checkpoints')

    def train_step(self, input_tensor, target_tensor):
        input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        pred_tensor = self.model(input_tensor)
        loss = self.loss_fn(pred_tensor, target_tensor)
        loss.backward()
        self.optimizer.step()
        return pred_tensor, loss.item()

    def test_step(self, input_tensor):
        self.model.eval()
        input_tensor = input_tensor.to(self.device)
        with torch.no_grad():
            pred_tensor = self.model(input_tensor)

        return pred_tensor

    def train(self):
        self.register_training()
        train_loader = self.dataset['train']
        val_loader = self.dataset['val']

        total_steps = self.config['learner']['general']['total_steps']
        log_train_info_steps = self.config['learner']['general']['log_train_info_steps']  # Frequency of logging
        valid_steps = self.config['learner']['general']['valid_steps']
        keep_ckpt_steps = self.config['learner']['general']['keep_ckpt_steps']

        step = self.step
        if step == 0:
            print('Initiating new training session from the scratch.')
        else:
            print(f'Resuming training from previously saved checkpoint at step {self.step}.')
        while step < total_steps:
            for input_tensor, target_tensor in train_loader:
                if step >= total_steps:
                    break
                pred, loss = self.train_step(input_tensor, target_tensor)

                # Logging training information
                if step % log_train_info_steps == 0:
                    self.summary.add_scalar('Training Loss', loss, step)
                    print(f'Step: {step}, Training Loss: {loss}')

                if step % keep_ckpt_steps == 0 and step != 0:
                    self.saver.save_checkpoint(step, val_metric=None)

                step += 1

                # Validation and logging validation information
                if step % valid_steps == 0:
                    val_loss, val_metric = self.perform_validation(val_loader)
                    self.saver.update_and_save(step, val_metric)
                    self.summary.add_scalar('Validation Loss', val_loss, step)
                    self.summary.add_scalar('Validation WS-PSNR', val_metric, step)
                    print(f'Step: {step}, Validation Loss: {val_loss}, Validation WS-PSNR: {val_metric}')

    def perform_validation(self, val_loader):
        with torch.no_grad():
            val_loss = 0.0
            self.metric_functions['ws-psnr'].reset()  # Reset the metric at the start of validation

            for input_tensor, target_tensor in val_loader:
                pred = self.test_step(input_tensor)

                target_tensor = target_tensor.to(self.device)

                val_loss += self.loss_fn(pred, target_tensor).item()

                # Update the metric for the current batch

                self.metric_functions['ws-psnr'].update(pred * 255, target_tensor * 255)
            # Calculate average loss and metric
            val_loss /= len(val_loader)
            val_metric = self.metric_functions['ws-psnr'].get_result()
            return val_loss, val_metric

    def test(self):
        # Load the checkpoint
        restore_path = self.config['learner']['saver']['restore']
        if restore_path is not None and restore_path != 'null':
            restore_full_path = Path(self.log_dir) / restore_path
            self.saver.load_checkpoint(restore_full_path)
        else:
            print("No checkpoint path provided for testing.")
            return

        # Set the model to evaluation mode
        self.model.eval()

        # Prepare the test data loader
        test_loader = self.dataset['test']

        ws_psnr_metric = WSPSNRMetric()

        # Folder to save upscaled images
        upscaled_dir = Path(self.log_dir) / "upscaled"
        upscaled_dir.mkdir(parents=True, exist_ok=True)

        first_batch = next(iter(test_loader))

        image_counter = 0
        
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Processing images"):
                if len(batch) == 3:
                    input_tensor, target_tensor, lr_paths = batch
                    target_tensor = target_tensor.to(self.device)
                else:
                    input_tensor, lr_paths = batch

                input_tensor = input_tensor.to(self.device)

                start.record()

                # Generate upscaled images
                upscaled_tensors = self.model(input_tensor)

                # Synchronize CUDA after computation and before stopping the timer
                end.record()

                torch.cuda.synchronize()

                time_taken = start.elapsed_time(end)
                _, H, W = upscaled_tensors[0].shape
                tqdm.write(f"Time taken for frame of size {H}x{W}: {time_taken/1000:.4f} seconds")

                if len(batch) == 3:
                    ws_psnr_metric.update(upscaled_tensors * 255, target_tensor * 255)

                for i, upscaled_tensor in enumerate(upscaled_tensors):
                    # Convert the tensor to an image format
                    output = upscaled_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    if output.ndim == 3:
                        # Convert CHW to HWC and RGB to BGR for compatibility with cv2
                        output = np.transpose(output, (1, 2, 0))  # CHW to HWC
                        output = output[..., ::-1]  # RGB to BGR

                    # Convert from float to uint8
                    output = (output * 255.0).round().astype(np.uint8)

                    # Construct the path for saving the upscaled image
                    path_parts = lr_paths[i].split(os.path.sep)
                    bitrate, video_name, im_name = path_parts[-3], path_parts[-2], path_parts[-1]

                    save_dir = Path(self.log_dir) / "upscaled" / bitrate / video_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    image_path = save_dir / im_name

                    # Save the image
                    cv2.imwrite(str(image_path), output)
                    image_counter += 1

        if len(first_batch) == 3:
            average_wspnsr = ws_psnr_metric.get_result()
            print(f"Average WS-PSNR: {average_wspnsr}")

        print(f"Upscaled images are saved in {upscaled_dir}")






























import torch
import os
from accelerate import Accelerator
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from learner.metrics import WSPSNRMetric
from learner.saver import Saver
from learner.loss import *
import numpy as np
import time
import cv2
from accelerate import DistributedDataParallelKwargs

class StandardLearner():
    """
        config: A `dict` contains the configuration of the learner.
        model: A list of `pytorch` objects which generate predictions.
        dataset: A dataset `dict` contains dataloader for different split.
        step: An `int` represents the current step. Initialize to 0.
        optimizer: A `tf.keras.optimizers` is used to optimize the model. Initialize to None.
        lr_scheduler: A `tf.keras.optimizers.schedules.LearningRateSchedule` is used to schedule
            the leaning rate. Initialize to None.
        metric_functions: A `dict` contains one or multiple functions which are used to
            metric the results. Initialize to {}.
        saver: A `Saver` is used to save checkpoints. Initialize to None.
        summary: A `TensorboardSummary` is used to save eventfiles.
        log_dir: A `str` represents the directory which records experiments.
        steps: An `int` represents the number of train steps.
        log_train_info_steps: An `int` represents frequency of logging training information.
        keep_ckpt_steps: An `int` represents frequency of saving checkpoint.
        valid_steps: An `int` represents frequency  of validation.
    """

    def __init__(self, config, model, dataset, log_dir):
        self.config = config
        self.model = model
        self.dataset = dataset
        self.log_dir = log_dir
        self.loss_fn = None
        self.step = 0
        self.optimizer = None
        self.lr_scheduler = None
        self.metric_functions = {}
        self.saver = None
        self.summary = SummaryWriter(log_dir)
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)

        self.accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])

        device = self.accelerator.device
        # Initialize training components
        self.model = model.to(device)
        self.register_training()

        # Check if a checkpoint path is provided in the config and load it
        restore_path = config['learner']['saver']['restore']
        if restore_path is not None and restore_path != 'null':
            restore_full_path = Path(log_dir) / restore_path
            self.saver = Saver(self.model, self.optimizer, save_dir=Path(log_dir) / 'checkpoints')
            self.saver.load_checkpoint(restore_full_path)
            self.step = self.saver.step

        # Set device
        #self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #self.model.to(self.device)

        # Set random seed for reproducibility
        torch.manual_seed(2454)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(2454)
            torch.cuda.manual_seed_all(2454)
        np.random.seed(2454)



    def register_training(self):
        # Extract optimizer configurations
        opt_config = self.config['learner']['optimizer']
        OptimizerClass = getattr(torch.optim, opt_config['name'])

        # Adjust the configuration
        betas = (opt_config.get('beta_1', 0.9), opt_config.get('beta_2', 0.999))
        opt_config_adjusted = {'lr': opt_config.get('lr', 0.001), 'betas': betas}

        # Initialize the optimizer with the adjusted configuration
        self.optimizer = OptimizerClass(self.model.parameters(), **opt_config_adjusted)
        print(self.optimizer.param_groups[0]['lr'])

        # Extract lr_scheduler configurations
        #lr_scheduler_config = self.config['learner']['lr_scheduler']
        #if lr_scheduler_config['name'] == 'ExponentialDecay':
            #self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                #self.optimizer,
                #gamma=lr_scheduler_config['decay_rate']
            #)
        self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 'max', factor=0.1, patience=5, verbose=True)
        # Other lr_schedulers can be added here

        # Define loss function
        loss_config = self.config.get('learner', {}).get('loss', {})
        loss_name = loss_config.get('name', 'MSELoss').lower()
        print(loss_name)

        if loss_name == 'charbonnierloss':
            self.loss_fn = CharbonnierLoss(**loss_config.get('params', {}))
        elif loss_name == 'perceptualloss':
            self.loss_fn = PerceptualLoss(**loss_config.get('params', {}))
        elif loss_name == 'ganloss':
            self.loss_fn = GANLoss(**loss_config.get('params', {}))
        elif loss_name == 'tvloss':
            self.loss_fn = TVLoss(**loss_config.get('params', {}))
        else:
            self.loss_fn = torch.nn.MSELoss()  # Default loss
        # Define metrics
        self.metric_functions['ws-psnr'] = WSPSNRMetric()

        if self.saver is None:  # Only initialize if not already set up by checkpoint loading
            self.saver = Saver(self.model, self.optimizer, save_dir=Path(self.log_dir) / 'checkpoints')

    def train_step(self, input_tensor, target_tensor):
        #input_tensor, target_tensor = input_tensor.to(self.device), target_tensor.to(self.device)
        self.model.train()
        self.optimizer.zero_grad()
        pred_tensor = self.model(input_tensor)
        loss = self.loss_fn(pred_tensor, target_tensor)
        self.accelerator.backward(loss)
        self.optimizer.step()
        return pred_tensor, loss.item()

    def test_step(self, input_tensor):
        self.model.eval()
        #input_tensor = input_tensor.to(self.device)

        with torch.no_grad():
            # Adjusting the input tensor to have dimensions compatible with the window size
            _, _, h_old, w_old = input_tensor.size()

            h_pad = (h_old // 32 + 1) * 32 - h_old
            w_pad = (w_old // 32 + 1) * 32 - w_old
            #input_tensor = input_tensor.to(self.device)
            input_tensor = torch.cat([input_tensor, torch.flip(input_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
            input_tensor = torch.cat([input_tensor, torch.flip(input_tensor, [3])], 3)[:, :, :, :w_old + w_pad]
            pred_tensor = self.model(input_tensor)
            pred_tensor = pred_tensor[..., :h_old *4, :w_old * 4]

        return pred_tensor


    def train(self):
        self.register_training()
        train_loader = self.dataset['train']
        val_loader = self.dataset['val']
        total_steps = self.config['learner']['general']['total_steps']
        log_train_info_steps = self.config['learner']['general']['log_train_info_steps']  # Frequency of logging
        valid_steps = self.config['learner']['general']['valid_steps']
        keep_ckpt_steps = self.config['learner']['general']['keep_ckpt_steps']
        train_loader, val_loader, self.model, self.optimizer, self.lr_scheduler = self.accelerator.prepare(train_loader, val_loader, self.model, self.optimizer, self.lr_scheduler)
 

        step = self.step
        if step == 0:
            print('Initiating new training session from the scratch.')
        else:
            print(f'Resuming training from previously saved checkpoint at step {self.step}.')
        while step < total_steps:
            for input_tensor, target_tensor in train_loader:
                if step >= total_steps:
                    break
                pred, loss = self.train_step(input_tensor, target_tensor)
                #self.accelerator.wait_for_everyone() 




                # Logging training information
                if step % log_train_info_steps == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    self.summary.add_scalar('Training Loss', loss, step)
                    self.summary.add_scalar('Learning Rate', current_lr, step)
                    print(f'Step: {step}, Training Loss: {loss}, Learning Rate: {current_lr}')
                if step % keep_ckpt_steps == 0 and step != 0:
                    self.saver.save_checkpoint(step, val_metric=None)

                step += 1

                # Validation and logging validation information
                if step % valid_steps == 0:
                    val_loss, val_metric = self.perform_validation(val_loader)
                    self.saver.update_and_save(step, val_metric)
                    self.summary.add_scalar('Validation Loss', val_loss, step)
                    self.summary.add_scalar('Validation WS-PSNR', val_metric, step)
                    #self.lr_scheduler.step(val_loss)
                    self.lr_scheduler.step(val_metric)
                    self.accelerator.wait_for_everyone()
                    print(f'Step: {step}, Validation Loss: {val_loss}, Validation WS-PSNR: {val_metric}')

    def perform_validation(self, val_loader):
        with torch.no_grad():
            val_loss = 0.0
            self.metric_functions['ws-psnr'].reset()  # Reset the metric at the start of validation

            for input_tensor, target_tensor in val_loader:
                pred = self.test_step(input_tensor)

                #target_tensor = target_tensor.to(self.device)

                val_loss += self.loss_fn(pred, target_tensor).item()

                # Update the metric for the current batch
                pred, target_tensor = self.accelerator.gather((pred, target_tensor))
                self.accelerator.wait_for_everyone()

                self.metric_functions['ws-psnr'].update(pred * 255, target_tensor * 255)
            # Calculate average loss and metric
            val_loss /= len(val_loader)
            val_metric = self.metric_functions['ws-psnr'].get_result()
            return val_loss, val_metric

    def test(self):
        # Load the checkpoint
        restore_path = self.config['learner']['saver']['restore']
        if restore_path is not None and restore_path != 'null':
            restore_full_path = Path(self.log_dir) / restore_path
            self.saver.load_checkpoint(restore_full_path)
        else:
            print("No checkpoint path provided for testing.")
            return

        # Set the model to evaluation mode
        self.model.eval()

        # Prepare the test data loader
        test_loader = self.dataset['test']

        ws_psnr_metric = WSPSNRMetric()

        # Folder to save upscaled images
        upscaled_dir = Path(self.log_dir) / "upscaled"
        upscaled_dir.mkdir(parents=True, exist_ok=True)

        first_batch = next(iter(test_loader))
        print(len(first_batch))

        image_counter = 0
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Processing images"):
                if len(batch) == 3:
                    input_tensor, target_tensor, lr_paths = batch
                    target_tensor = target_tensor.to('cuda')
                else:
                    input_tensor, lr_paths = batch

                input_tensor = input_tensor.to('cuda')

                start.record()

                # Generate upscaled images
                upscaled_tensors = self.test_step(input_tensor)

                # Synchronize CUDA after computation and before stopping the timer
                end.record()

                torch.cuda.synchronize()

                time_taken = start.elapsed_time(end)
                _, H, W = upscaled_tensors[0].shape
                tqdm.write(f"Time taken for frame of size {H}x{W}: {time_taken/1000:.4f} seconds")

                if len(batch) == 3:
                    ws_psnr_metric.update(upscaled_tensors * 255, target_tensor * 255)

                for i, upscaled_tensor in enumerate(upscaled_tensors):
                    # Convert the tensor to an image format
                    output = upscaled_tensor.data.squeeze().float().cpu().clamp_(0, 1).numpy()
                    if output.ndim == 3:
                        # Convert CHW to HWC and RGB to BGR for compatibility with cv2
                        output = np.transpose(output, (1, 2, 0))  # CHW to HWC
                        output = output[..., ::-1]  # RGB to BGR

                    # Convert from float to uint8
                    output = (output * 255.0).round().astype(np.uint8)

                    # Construct the path for saving the upscaled image
                    path_parts = lr_paths[i].split(os.path.sep)
                    bitrate, video_name, im_name = path_parts[-3], path_parts[-2], path_parts[-1]

                    save_dir = Path(self.log_dir) / "upscaled" / bitrate / video_name
                    save_dir.mkdir(parents=True, exist_ok=True)
                    image_path = save_dir / im_name

                    # Save the image
                    cv2.imwrite(str(image_path), output)
                    image_counter += 1

        if len(first_batch) == 3:
            average_wspnsr = ws_psnr_metric.get_result()
            print(f"Average WS-PSNR: {average_wspnsr}")

        print(f"Upscaled images are saved in {upscaled_dir}")
