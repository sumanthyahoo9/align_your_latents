# tests/test_time_embedding.py
import torch
from src.modules.time_embedding import TimeEmbedding

def test_time_embedding_shape():
    """Test output shape"""
    batch_size = 4
    dim = 1280
    
    time_emb = TimeEmbedding(dim)
    t = torch.randint(0, 1000, (batch_size,))
    
    emb = time_emb(t)
    
    assert emb.shape == (batch_size, dim)

def test_time_embedding_different_timesteps():
    """Different timesteps should give different embeddings"""
    time_emb = TimeEmbedding(1280)
    
    t1 = torch.tensor([0])
    t2 = torch.tensor([500])
    
    emb1 = time_emb(t1)
    emb2 = time_emb(t2)
    
    # Should be different!
    assert not torch.allclose(emb1, emb2)

def test_time_embedding_deterministic():
    """Same timestep should give same embedding"""
    time_emb = TimeEmbedding(1280)
    
    t = torch.tensor([42])
    
    emb1 = time_emb(t)
    emb2 = time_emb(t)
    
    assert torch.allclose(emb1, emb2)