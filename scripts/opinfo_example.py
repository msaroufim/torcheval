#!/usr/bin/env python3
"""
Complete OpInfo Explorer - Comprehensive analysis of PyTorch OpInfo tests

This script provides a complete view of all available OpInfo operations,
their input shapes, distributions, and test configurations.
"""

import torch
from torch.testing._internal.opinfo.core import OpInfo
from torch.testing._internal.common_methods_invocations import op_db
import traceback
from collections import defaultdict

def main():
    print("=" * 80)
    print("PYTORCH OPINFO COMPREHENSIVE EXPLORER")
    print("=" * 80)
    
    # 1. Overview of all operations
    print(f"\n📊 OVERVIEW")
    print(f"Total operations in op_db: {len(op_db)}")
    print(f"PyTorch version: {torch.__version__}")
    
    # 2. List all operation names
    print(f"\n📋 ALL AVAILABLE OPERATIONS")
    op_names = sorted([op.name for op in op_db])
    for i, name in enumerate(op_names, 1):
        print(f"{i:3d}. {name}")
    
    # 3. Categorize operations by type
    print(f"\n🏷️  OPERATION CATEGORIES")
    categories = defaultdict(list)
    for op in op_db:
        # Simple categorization based on name patterns
        name = op.name.lower()
        if any(x in name for x in ['conv', 'pool', 'norm']):
            categories['Neural Network'].append(op.name)
        elif any(x in name for x in ['add', 'sub', 'mul', 'div', 'pow']):
            categories['Arithmetic'].append(op.name)
        elif any(x in name for x in ['sin', 'cos', 'exp', 'log', 'sqrt']):
            categories['Mathematical'].append(op.name)
        elif any(x in name for x in ['sum', 'mean', 'max', 'min']):
            categories['Reduction'].append(op.name)
        elif any(x in name for x in ['view', 'reshape', 'transpose', 'permute']):
            categories['Shape Manipulation'].append(op.name)
        else:
            categories['Other'].append(op.name)
    
    for category, ops in categories.items():
        print(f"  {category}: {len(ops)} operations")
        print(f"    {', '.join(sorted(ops)[:5])}{'...' if len(ops) > 5 else ''}")
    
    # 4. Detailed analysis of selected operations
    print(f"\n🔍 DETAILED ANALYSIS OF KEY OPERATIONS")
    key_operations = ['add', 'conv2d', 'relu', 'softmax', 'matmul', 'sum']
    
    for op_name in key_operations:
        analyze_operation_detailed(op_name)
    
    # 5. Statistical summary across all operations
    print(f"\n📈 STATISTICAL SUMMARY ACROSS ALL OPERATIONS")
    analyze_all_operations_stats()
    
    # 6. Edge cases and special inputs exploration
    print(f"\n⚠️  EDGE CASES AND SPECIAL INPUTS")
    explore_edge_cases(['sigmoid', 'log', 'div'])

def analyze_operation_detailed(op_name):
    """Detailed analysis of a specific operation"""
    print(f"\n--- {op_name.upper()} ---")
    
    op_info = next((op for op in op_db if op.name == op_name), None)
    if not op_info:
        print(f"❌ Operation '{op_name}' not found")
        return
    
    try:
        # Basic info
        print(f"  Aliases: {op_info.aliases}")
        print(f"  Supported dtypes: {op_info.dtypes}")
        print(f"  Supports autograd: {op_info.supports_autograd}")
        print(f"  Supports sparse: {getattr(op_info, 'supports_sparse', 'Unknown')}")
        print(f"  Supports complex: {getattr(op_info, 'supports_complex', 'Unknown')}")
        
        # Generate and analyze sample inputs
        try:
            samples = list(op_info.sample_inputs(device='cpu', dtype=torch.float32))
            print(f"  Number of test samples: {len(samples)}")
            
            # Analyze shapes and distributions
            shapes = []
            for i, sample in enumerate(samples[:5]):  # Look at first 5 samples
                tensor = sample.input
                shapes.append(tensor.shape)
                
                print(f"    Sample {i+1}:")
                print(f"      Shape: {tensor.shape}")
                print(f"      Data range: [{tensor.min().item():.4f}, {tensor.max().item():.4f}]")
                print(f"      Mean±Std: {tensor.mean().item():.4f}±{tensor.std().item():.4f}")
                print(f"      Args: {sample.args}")
                print(f"      Kwargs: {sample.kwargs}")
                
                # Check for special values
                special_vals = []
                if torch.isnan(tensor).any(): special_vals.append("NaN")
                if torch.isinf(tensor).any(): special_vals.append("Inf")
                if (tensor == 0).any(): special_vals.append("Zeros")
                if special_vals:
                    print(f"      Special values: {', '.join(special_vals)}")
            
            # Shape analysis
            unique_shapes = list(set(shapes))
            print(f"  Unique input shapes tested: {len(unique_shapes)}")
            if len(unique_shapes) <= 10:
                print(f"    Shapes: {unique_shapes}")
        
        except Exception as e:
            print(f"  ❌ Error generating samples: {str(e)}")
            
    except Exception as e:
        print(f"  ❌ Error analyzing operation: {str(e)}")

def analyze_all_operations_stats():
    """Statistical analysis across all operations"""
    stats = {
        'total_ops': len(op_db),
        'supports_autograd': 0,
        'dtype_support': defaultdict(int),
        'sample_counts': [],
        'common_shapes': defaultdict(int),
        'error_count': 0
    }
    
    for op_info in op_db:
        try:
            if op_info.supports_autograd:
                stats['supports_autograd'] += 1
            
            # Count dtype support
            for dtype in op_info.dtypes:
                stats['dtype_support'][str(dtype)] += 1
            
            # Sample analysis (limit to avoid slowdown)
            try:
                samples = list(op_info.sample_inputs(device='cpu', dtype=torch.float32))
                stats['sample_counts'].append(len(samples))
                
                # Shape analysis (first sample only for speed)
                if samples:
                    shape = samples[0].input.shape
                    stats['common_shapes'][len(shape)] += 1  # Track by dimensionality
            except:
                pass
                
        except Exception:
            stats['error_count'] += 1
    
    # Print statistics
    print(f"  Operations supporting autograd: {stats['supports_autograd']}/{stats['total_ops']} ({100*stats['supports_autograd']/stats['total_ops']:.1f}%)")
    print(f"  Operations with analysis errors: {stats['error_count']}")
    
    if stats['sample_counts']:
        avg_samples = sum(stats['sample_counts']) / len(stats['sample_counts'])
        print(f"  Average test samples per operation: {avg_samples:.1f}")
        print(f"  Range of test samples: {min(stats['sample_counts'])}-{max(stats['sample_counts'])}")
    
    print("  Most common tensor dimensionalities:")
    for dims, count in sorted(stats['common_shapes'].items(), key=lambda x: x[1], reverse=True)[:5]:
        print(f"    {dims}D tensors: {count} operations")
    
    print("  Most supported dtypes:")
    for dtype, count in sorted(stats['dtype_support'].items(), key=lambda x: x[1], reverse=True)[:8]:
        print(f"    {dtype}: {count} operations")

def explore_edge_cases(op_names):
    """Explore edge cases and special input values"""
    for op_name in op_names:
        print(f"\n--- EDGE CASES: {op_name.upper()} ---")
        
        op_info = next((op for op in op_db if op.name == op_name), None)
        if not op_info:
            continue
            
        try:
            samples = list(op_info.sample_inputs(device='cpu', dtype=torch.float32))
            
            edge_case_count = 0
            for sample in samples:
                tensor = sample.input
                
                # Check for various edge cases
                has_nan = torch.isnan(tensor).any()
                has_inf = torch.isinf(tensor).any()
                has_zero = (tensor == 0).any()
                has_negative = (tensor < 0).any()
                is_very_small = (tensor.abs() < 1e-6).any()
                is_very_large = (tensor.abs() > 1e6).any()
                
                edge_cases = []
                if has_nan: edge_cases.append("NaN")
                if has_inf: edge_cases.append("Inf")
                if has_zero: edge_cases.append("Zero")
                if has_negative: edge_cases.append("Negative")
                if is_very_small: edge_cases.append("Very small")
                if is_very_large: edge_cases.append("Very large")
                
                if edge_cases:
                    edge_case_count += 1
                    print(f"  Sample with edge cases: {', '.join(edge_cases)}")
                    print(f"    Shape: {tensor.shape}, Range: [{tensor.min().item():.6f}, {tensor.max().item():.6f}]")
            
            print(f"  Total samples with edge cases: {edge_case_count}/{len(samples)}")
            
        except Exception as e:
            print(f"  ❌ Error exploring edge cases: {str(e)}")

def find_operations_by_keyword(keyword):
    """Helper function to find operations containing a keyword"""
    matching_ops = [op.name for op in op_db if keyword.lower() in op.name.lower()]
    return matching_ops

analyze_all_operations_stats()
