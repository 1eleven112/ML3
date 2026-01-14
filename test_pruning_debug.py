"""
Debug script to test pruning mask preservation
"""
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import tempfile

# Test 1: Basic pruning behavior
print("=" * 50)
print("Test 1: Basic PyTorch Pruning")
print("=" * 50)

layer = nn.Linear(100, 50)
print(f"Before pruning:")
print(f"  Non-zero weights: {torch.count_nonzero(layer.weight).item()}")
print(f"  Has weight_orig: {hasattr(layer, 'weight_orig')}")
print(f"  Has weight_mask: {hasattr(layer, 'weight_mask')}")

# Apply pruning
prune.l1_unstructured(layer, name='weight', amount=0.5)
print(f"\nAfter applying pruning (with mask):")
print(f"  Non-zero weights: {torch.count_nonzero(layer.weight).item()}")
print(f"  Has weight_orig: {hasattr(layer, 'weight_orig')}")
print(f"  Has weight_mask: {hasattr(layer, 'weight_mask')}")

# Test state_dict
print(f"\nState dict keys: {list(layer.state_dict().keys())}")

# Save and load state_dict
with tempfile.TemporaryDirectory() as tmpdir:
    save_path = os.path.join(tmpdir, 'layer.pt')
    torch.save(layer.state_dict(), save_path)
    
    # Create new layer and load
    new_layer = nn.Linear(100, 50)
    new_layer.load_state_dict(torch.load(save_path))
    
    print(f"\nAfter loading state_dict:")
    print(f"  Non-zero weights: {torch.count_nonzero(new_layer.weight).item()}")
    print(f"  Has weight_orig: {hasattr(new_layer, 'weight_orig')}")
    print(f"  Has weight_mask: {hasattr(new_layer, 'weight_mask')}")
    
# Remove pruning
prune.remove(layer, 'weight')
print(f"\nAfter prune.remove():")
print(f"  Non-zero weights: {torch.count_nonzero(layer.weight).item()}")
print(f"  Has weight_orig: {hasattr(layer, 'weight_orig')}")
print(f"  Has weight_mask: {hasattr(layer, 'weight_mask')}")

# Test 2: With BERT model
print("\n" + "=" * 50)
print("Test 2: BERT Model Pruning")
print("=" * 50)

try:
    from transformers import BertForSequenceClassification
    from src.pruning import BERTPruner, make_pruning_permanent
    from src.utils import count_parameters
    
    # Load a small BERT model
    print("\nLoading BERT model...")
    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
    
    print("\nBefore pruning:")
    params_before = count_parameters(model)
    print(f"  Total params: {params_before['total']:,}")
    print(f"  Non-zero params: {params_before['non_zero']:,}")
    print(f"  Sparsity: {params_before['sparsity']:.2%}")
    
    # Apply pruning
    print("\nApplying 50% pruning...")
    pruner = BERTPruner(model)
    pruner.apply_magnitude_pruning(sparsity_per_layer=0.5)
    
    print("\nAfter pruning (with masks):")
    params_after_prune = count_parameters(model)
    print(f"  Total params: {params_after_prune['total']:,}")
    print(f"  Non-zero params: {params_after_prune['non_zero']:,}")
    print(f"  Sparsity: {params_after_prune['sparsity']:.2%}")
    
    # Check a specific layer
    sample_layer = model.bert.encoder.layer[0].attention.self.query
    print(f"\nSample layer (encoder.layer[0].attention.self.query):")
    print(f"  Has weight_orig: {hasattr(sample_layer, 'weight_orig')}")
    print(f"  Has weight_mask: {hasattr(sample_layer, 'weight_mask')}")
    if hasattr(sample_layer, 'weight'):
        print(f"  Non-zero weights: {torch.count_nonzero(sample_layer.weight).item()}")
    
    # Save and load state_dict
    print("\nTesting state_dict save/load...")
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, 'model.pt')
        torch.save(model.state_dict(), save_path)
        
        # Load back
        model.load_state_dict(torch.load(save_path))
        
        print("After loading state_dict:")
        params_after_load = count_parameters(model)
        print(f"  Total params: {params_after_load['total']:,}")
        print(f"  Non-zero params: {params_after_load['non_zero']:,}")
        print(f"  Sparsity: {params_after_load['sparsity']:.2%}")
        
        # Check sample layer again
        print(f"\nSample layer after load:")
        print(f"  Has weight_orig: {hasattr(sample_layer, 'weight_orig')}")
        print(f"  Has weight_mask: {hasattr(sample_layer, 'weight_mask')}")
    
    # Make pruning permanent
    print("\nMaking pruning permanent...")
    make_pruning_permanent(model)
    
    print("\nAfter make_pruning_permanent:")
    params_after_permanent = count_parameters(model)
    print(f"  Total params: {params_after_permanent['total']:,}")
    print(f"  Non-zero params: {params_after_permanent['non_zero']:,}")
    print(f"  Sparsity: {params_after_permanent['sparsity']:.2%}")
    
    # Check sample layer again
    print(f"\nSample layer after permanent:")
    print(f"  Has weight_orig: {hasattr(sample_layer, 'weight_orig')}")
    print(f"  Has weight_mask: {hasattr(sample_layer, 'weight_mask')}")
    print(f"  Non-zero weights: {torch.count_nonzero(sample_layer.weight).item()}")

except Exception as e:
    print(f"Error in BERT test: {e}")
    import traceback
    traceback.print_exc()
