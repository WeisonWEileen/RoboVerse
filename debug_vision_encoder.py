#!/usr/bin/env python3
"""
Debug script for vision_encoder zero output issue
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from humanoid_visualrl.actor_critic.actor_critic_cnn import ActorCriticCNN

def debug_vision_encoder():
    """Debug the vision encoder step by step"""
    
    print("🔍 Debugging Vision Encoder Zero Output Issue")
    print("=" * 60)
    
    # Create a dummy model to get the vision encoder
    print("1. Creating ActorCriticCNN model...")
    model = ActorCriticCNN(
        num_actor_obs=100,
        num_critic_obs=100, 
        num_actions=12
    )
    
    vision_encoder = model.vision_encoder
    print(f"Vision encoder: {vision_encoder}")
    
    # Test with different types of input data
    test_cases = [
        ("zeros", torch.zeros(1, 3, 48, 64)),
        ("ones", torch.ones(1, 3, 48, 64)),
        ("random_0_1", torch.rand(1, 3, 48, 64)),
        ("random_imagenet", torch.rand(1, 3, 48, 64) * 255.0 / 255.0),
        ("gaussian", torch.randn(1, 3, 48, 64) * 0.5 + 0.5),
    ]
    
    print("\n2. Testing with different input types:")
    print("-" * 40)
    
    for name, test_input in test_cases:
        print(f"\n📊 Testing with {name} input...")
        print(f"Input stats: min={test_input.min():.4f}, max={test_input.max():.4f}, mean={test_input.mean():.4f}")
        
        # Forward pass through each layer
        x = test_input
        for i, layer in enumerate(vision_encoder):
            x_prev = x
            x = layer(x)
            
            if isinstance(layer, nn.Conv2d):
                print(f"  Conv2d layer {i}: {x_prev.shape} -> {x.shape}")
                print(f"    Output stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}")
                print(f"    Zero ratio: {(x == 0).float().mean():.4f}")
                
                # Check if all outputs are zero
                if torch.all(x == 0):
                    print(f"    ❌ ALL OUTPUTS ARE ZERO at layer {i}!")
                    print(f"    Weight stats: min={layer.weight.min():.4f}, max={layer.weight.max():.4f}")
                    print(f"    Bias stats: min={layer.bias.min():.4f}, max={layer.bias.max():.4f}")
                    break
            elif isinstance(layer, nn.ReLU):
                print(f"  ReLU layer {i}: activated {(x > 0).float().mean():.4f} of neurons")
                if torch.all(x == 0):
                    print(f"    ❌ ReLU killed all activations!")
            elif isinstance(layer, nn.Flatten):
                print(f"  Flatten layer {i}: {x_prev.shape} -> {x.shape}")
                
        final_output = x
        print(f"Final output stats: min={final_output.min():.4f}, max={final_output.max():.4f}, mean={final_output.mean():.4f}")
        print(f"Final output shape: {final_output.shape}")
        print(f"All zeros? {torch.all(final_output == 0)}")

def analyze_weight_initialization():
    """Analyze the weight initialization of the vision encoder"""
    print("\n3. Analyzing weight initialization:")
    print("-" * 40)
    
    model = ActorCriticCNN(
        num_actor_obs=100,
        num_critic_obs=100,
        num_actions=12
    )
    
    for i, layer in enumerate(model.vision_encoder):
        if isinstance(layer, nn.Conv2d):
            print(f"\nConv2d layer {i}:")
            print(f"  Weight shape: {layer.weight.shape}")
            print(f"  Weight stats: min={layer.weight.min():.6f}, max={layer.weight.max():.6f}, mean={layer.weight.mean():.6f}, std={layer.weight.std():.6f}")
            if layer.bias is not None:
                print(f"  Bias stats: min={layer.bias.min():.6f}, max={layer.bias.max():.6f}, mean={layer.bias.mean():.6f}")

def test_gradient_flow():
    """Test if gradients can flow through the network"""
    print("\n4. Testing gradient flow:")
    print("-" * 40)
    
    model = ActorCriticCNN(
        num_actor_obs=100,
        num_critic_obs=100,
        num_actions=12
    )
    
    # Create dummy input
    vision_input = torch.rand(4, 3, 48, 64, requires_grad=True)
    state_input = torch.rand(4, 100)
    
    # Forward pass
    vision_features = model.vision_encoder(vision_input)
    
    # Create a dummy loss
    loss = vision_features.sum()
    
    # Backward pass
    loss.backward()
    
    print(f"Vision input gradient norm: {vision_input.grad.norm():.6f}")
    
    # Check gradients for each layer
    for name, param in model.vision_encoder.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm()
            print(f"  {name}: grad_norm={grad_norm:.6f}")
            if grad_norm == 0:
                print(f"    ❌ Zero gradients detected!")
        else:
            print(f"  {name}: No gradients")

def suggest_fixes():
    """Suggest potential fixes for the zero output issue"""
    print("\n5. Suggested Fixes:")
    print("-" * 40)
    
    fixes = [
        "🔧 Check input data: Ensure vision_buf contains non-zero image data",
        "🔧 Verify normalization: Input should be in [0,1] range, not [-1,1]", 
        "🔧 Add proper weight initialization (Xavier/He initialization)",
        "🔧 Consider using LeakyReLU or ELU instead of ReLU to avoid dead neurons",
        "🔧 Add BatchNorm layers to stabilize training",
        "🔧 Reduce stride or kernel size to preserve spatial information",
        "🔧 Add residual connections for better gradient flow",
        "🔧 Check learning rate - too high can cause gradient explosion",
        "🔧 Verify that model is in training mode during training",
        "🔧 Add gradient clipping to prevent gradient explosion"
    ]
    
    for fix in fixes:
        print(f"  {fix}")

if __name__ == "__main__":
    debug_vision_encoder()
    analyze_weight_initialization()
    test_gradient_flow()
    suggest_fixes()
    
    print("\n" + "=" * 60)
    print("🎯 Debug complete! Check the output above for specific issues.")

