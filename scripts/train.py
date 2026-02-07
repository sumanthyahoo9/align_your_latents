"""
Script to train the Video LDM model
"""
import torch
from src.training.trainer import VideoLDMTrainer
from src.training.config import TrainingConfig
def main():
    """
    Main entry point
    """
    config = TrainingConfig(
        num_epochs=10,
        batch_size=2,
        learning_rate=1e-4,
        num_frames=8,
        device="cuda" if torch.cuda.is_available() else "cpu",
        checkpoint_dir="./checkpoints",
        log_dir="./logs"
    )
    # Initialize trainer
    trainer = VideoLDMTrainer(config)
    
    # TODO: Create dataloader
    # For now, dummy data for testing
    dummy_data = [torch.randn(config.batch_size, 4, config.num_frames, 64, 64) for _ in range(10)]
    
    # Train
    trainer.train(dummy_data)
    
    print("✓ Training complete!")