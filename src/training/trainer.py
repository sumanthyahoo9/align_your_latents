"""
The main trainer class
"""
import os
from pathlib import Path
from tqdm import tqdm
import torch
import torch.optim as optim
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter
from src.training.diffusion import DDPMScheduler
from src.training.config import TrainingConfig
from src.models.video_ldm import VideoUnet

class VideoLDMTrainer:
    """
    Trainer for Video Diffusion models
    """
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        # Create directories
        Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(config.log_dir).mkdir(parents=True, exist_ok=True)
        # Initialize the model
        self.model = VideoUnet(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            base_channels=config.base_channels,
            channel_mult=config.channel_mult,
            num_res_blocks=config.num_res_blocks,
            attention_resolutions=config.attention_resolutions,
            input_resolution=config.input_resolution,
            temporal_layer_type=config.temporal_layer_type,
            temporal_attention_heads=config.temporal_attention_heads,
            add_temporal_at_resolutions=config.add_temporal_at_resolutions
        ).to(self.device)
        # Initialize diffusion scheduler
        self.scheduler = DDPMScheduler(
            num_timesteps=config.num_timesteps,
            beta_start=config.beta_start,
            beta_end=config.beta_end,
            device=self.device
        )
        # Initialize optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
        # Mixed precision training
        self.scaler = GradScaler() if config.mixed_precision else None
        
        # Logging
        self.writer = SummaryWriter(log_dir=config.log_dir)
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        
        print(f"✓ Trainer initialized on {self.device}")
        print(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
    
    def train_step(self, batch: torch.Tensor) -> float:
        """
        Single training step
        batch: Video latents [B, C, T, H, W]
        loss: training loss value
        """
        self.model.train()
        # move batch to device
        x_0 = batch.to(self.device)
        B = x_0.shape[0]
        # Sample random timesteps
        timesteps = self.scheduler.sample_timesteps(B)
        # Sample noise
        noise = torch.randn_like(x_0)
        # Add noise to get x_t
        x_t = self.scheduler.add_noise(x_0, noise, timesteps)
        
        # Forward pass with mixed precision
        if self.scaler is not None:
            with autocast():
                # Predict the noise
                noise_pred = self.model(x_t, timesteps)
                # MSE loss
                loss = nn.functional.mse_loss(noise_pred, noise)
        else:
            # Predict noise
            noise_pred = self.model(x_t, timesteps)
            # MSE loss
            loss = nn.functional.mse_loss(noise_pred, noise)
        
        # Backward pass
        self.optimizer.zero_grad()
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            # Gradient clipping
            if self.config.gradient_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.gradient_clip
                )
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            # Gradient clipping
            if self.config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config.gradient_clip
                )
            self.optimizer.step()
        return loss.item()
    
    def train(self, dataloader):
        """
        Main trainin loop
        """
        for epoch in range(self.config.num_epochs):
            self.epoch = epoch
            epoch_loss = 0.0
            # Progress bar
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
            for _, batch in enumerate(pbar):
                # Single training step
                loss = self.train_step(batch)
                epoch_loss += loss
                # Logging
                if self.global_step % self.config.log_every == 0:
                    self.writer.add_scalar("train/loss", loss, self.global_step)
                    pbar.set_postfix({"loss": f"{loss:.4f}", "step": self.global_step})
                # Checkpointing
                if self.global_step % self.config.save_every == 0 and self.global_step > 0:
                    self.save_checkpoint(f"Checkpoint_step_{self.global_step}.pt")
                self.global_step += 1
            # Epoch statistics
            avg_epoch_loss = epoch_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed. Avg loss: {avg_epoch_loss:.4f}")
            self.writer.add_scalar("train/epoch_loss", avg_epoch_loss, epoch)
            
            # Save epoch checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        print("Training complete")
        self.writer.close()
    
    def save_checkpoint(self, filename: str) -> None:
        """
        Save the checkpoint
        """
        checkpoint_path = os.path.join(self.config.checkpoint_dir, filename)
        checkpoint = {
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
        }
        if self.scaler is not None:
            checkpoint['scaler_state_dict'] = self.scaler.state_dict()
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """
        Load model checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # Restore model state
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Restore training state
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        # Restore scaler if using mixed precision
        if self.scaler is not None and 'scaler_state_dict' in checkpoint:
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        print(f"✓ Checkpoint loaded: {checkpoint_path}")
        print(f"  Resuming from epoch {self.epoch}, step {self.global_step}")
        


