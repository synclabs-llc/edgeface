"""
Complete EdgeFace to TFLite Converter
Handles LoRA weight merging, quantization removal, and full conversion pipeline.
"""

import torch
import torch.nn as nn
import os
import copy
from typing import Dict, Any, Optional

class LoRaLin(nn.Module):
    """LoRA Linear layer implementation from EdgeFace"""
    def __init__(self, in_features, out_features, rank, bias=True):
        super(LoRaLin, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=bias)

    def forward(self, input):
        x = self.linear1(input)
        x = self.linear2(x)
        return x

class EdgeFaceConverter:
    """Complete converter for EdgeFace models to TFLite"""
    
    def __init__(self):
        self.supported_models = {
            'edgeface_xs_gamma_06': {'has_lora': True, 'has_quantization': False},
            'edgeface_xs_q': {'has_lora': False, 'has_quantization': True},
            'edgeface_xxs': {'has_lora': False, 'has_quantization': False},
            'edgeface_base': {'has_lora': False, 'has_quantization': False},
            'edgeface_xxs_q': {'has_lora': False, 'has_quantization': True},
            'edgeface_s_gamma_05': {'has_lora': True, 'has_quantization': False},
        }
    
    def merge_lora_weights(self, model: nn.Module) -> nn.Module:
        """
        Merge LoRA weights back into standard Linear layers for ONNX export.
        
        Args:
            model: EdgeFace model with LoRaLin layers
            
        Returns:
            Model with merged weights as standard Linear layers
        """
        
        def _merge_lora_layer(lora_layer):
            """Merge a single LoRaLin layer back to standard Linear layer"""
            if isinstance(lora_layer, LoRaLin):
                # Get weights from the two linear layers
                w1 = lora_layer.linear1.weight  # Shape: (rank, in_features)
                w2 = lora_layer.linear2.weight  # Shape: (out_features, rank)
                
                # Merge: W_merged = W2 @ W1
                merged_weight = w2 @ w1  # Shape: (out_features, in_features)
                
                # Create new Linear layer
                in_features = lora_layer.in_features
                out_features = lora_layer.out_features
                has_bias = lora_layer.linear2.bias is not None
                
                new_layer = nn.Linear(in_features, out_features, bias=has_bias)
                new_layer.weight.data = merged_weight
                
                if has_bias:
                    new_layer.bias.data = lora_layer.linear2.bias.data
                    
                return new_layer
            
            return lora_layer
        
        # Create a deep copy of the model
        merged_model = copy.deepcopy(model)
        
        # Replace LoRaLin layers with merged Linear layers
        def replace_lora_recursive(module):
            for name, child in module.named_children():
                if isinstance(child, LoRaLin):
                    # Replace with merged layer
                    merged_layer = _merge_lora_layer(child)
                    setattr(module, name, merged_layer)
                else:
                    # Recursively process child modules
                    replace_lora_recursive(child)
        
        replace_lora_recursive(merged_model)
        return merged_model
    
    def get_exportable_model(self, model_name: str, original_get_model_func) -> nn.Module:
        """
        Get EdgeFace model ready for ONNX export.
        
        Args:
            model_name: EdgeFace model name
            original_get_model_func: Original get_model function from EdgeFace
            
        Returns:
            Model ready for ONNX export
        """
        
        if model_name not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_name}")
        
        model_info = self.supported_models[model_name]
        
        # Handle quantized models - get base version instead
        if model_info['has_quantization']:
            print(f"Removing quantization for ONNX export: {model_name}")
            base_name = model_name.replace('_q', '')
            if base_name in self.supported_models:
                model = original_get_model_func(base_name)
            else:
                # Manually create base model
                if model_name == 'edgeface_xs_q':
                    from .timmfr import get_timmfrv2
                    model = get_timmfrv2('edgenext_x_small', batchnorm=False)
                elif model_name == 'edgeface_xxs_q':
                    from .timmfr import get_timmfrv2
                    model = get_timmfrv2('edgenext_xx_small', batchnorm=False)
                else:
                    raise ValueError(f"Cannot create base model for {model_name}")
        else:
            # Get original model
            model = original_get_model_func(model_name)
        
        # Handle LoRA models - merge weights
        if model_info['has_lora']:
            print(f"Merging LoRA weights for model: {model_name}")
            model = self.merge_lora_weights(model)
        
        return model
    
    def export_to_onnx(self, model_name: str, original_get_model_func, 
                      output_path: str, input_shape: tuple = (1, 3, 112, 112)):
        """
        Export EdgeFace model to ONNX format.
        
        Args:
            model_name: EdgeFace model name
            original_get_model_func: Original get_model function
            output_path: Path to save ONNX model
            input_shape: Input tensor shape (batch, channels, height, width)
        """
        
        # Get model ready for export
        model = self.get_exportable_model(model_name, original_get_model_func)
        model.eval()
        
        # Create dummy input
        dummy_input = torch.randn(input_shape)
        
        # Export to ONNX
        torch.onnx.export(
            model,
            dummy_input,
            output_path,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            },
            opset_version=11,
            do_constant_folding=True,
            verbose=False,
            export_params=True
        )
        
        print(f"✓ ONNX model exported to: {output_path}")
        
        # Verify ONNX model
        try:
            import onnx
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            print("✓ ONNX model verification passed")
        except ImportError:
            print("! ONNX verification skipped (onnx package not installed)")
        except Exception as e:
            print(f"! ONNX verification failed: {e}")
    
    def convert_to_tflite(self, model_name: str, original_get_model_func,
                         output_dir: str = "./converted_models", 
                         quantize: bool = True,
                         input_shape: tuple = (1, 3, 112, 112)):
        """
        Complete conversion pipeline from EdgeFace to TFLite.
        
        Args:
            model_name: EdgeFace model name
            original_get_model_func: Original get_model function
            output_dir: Directory to save converted models
            quantize: Whether to apply quantization in TFLite
            input_shape: Input tensor shape
        """
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Step 1: Export to ONNX
        onnx_path = os.path.join(output_dir, f"{model_name}.onnx")
        self.export_to_onnx(model_name, original_get_model_func, onnx_path, input_shape)
        
        try:
            # Step 2: Convert ONNX to TensorFlow
            import onnx
            from onnx_tf.backend import prepare
            
            print("Converting ONNX to TensorFlow...")
            onnx_model = onnx.load(onnx_path)
            tf_rep = prepare(onnx_model)
            
            tf_model_path = os.path.join(output_dir, f"{model_name}_tf")
            tf_rep.export_graph(tf_model_path)
            print(f"✓ TensorFlow model saved to: {tf_model_path}")
            
            # Step 3: Convert TensorFlow to TFLite
            import tensorflow as tf
            
            print("Converting TensorFlow to TFLite...")
            converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
            
            if quantize:
                # Enable quantization
                converter.optimizations = [tf.lite.Optimize.DEFAULT]
                print("✓ Quantization enabled")
            
            # Optional: Set representative dataset for better quantization
            # converter.representative_dataset = self._representative_dataset_gen
            
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = os.path.join(output_dir, f"{model_name}_quantized.tflite" if quantize else f"{model_name}.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✓ TFLite model saved to: {tflite_path}")
            
            # Print model info
            interpreter = tf.lite.Interpreter(model_path=tflite_path)
            interpreter.allocate_tensors()
            
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            print(f"\nModel Information:")
            print(f"  Input shape: {input_details[0]['shape']}")
            print(f"  Output shape: {output_details[0]['shape']}")
            print(f"  Input dtype: {input_details[0]['dtype']}")
            print(f"  Output dtype: {output_details[0]['dtype']}")
            
            return tflite_path
            
        except ImportError as e:
            print(f"✗ Conversion failed: Missing dependencies - {e}")
            print("Install with: pip install onnx onnx-tf tensorflow")
            return None
        except Exception as e:
            print(f"✗ Conversion failed: {e}")
            return None
    
    def batch_convert(self, model_names: list, original_get_model_func, 
                     output_dir: str = "./converted_models"):
        """
        Convert multiple EdgeFace models to TFLite.
        
        Args:
            model_names: List of model names to convert
            original_get_model_func: Original get_model function
            output_dir: Output directory
        """
        
        results = {}
        
        for model_name in model_names:
            print(f"\n{'='*60}")
            print(f"Converting {model_name}...")
            print(f"{'='*60}")
            
            try:
                tflite_path = self.convert_to_tflite(
                    model_name, original_get_model_func, output_dir
                )
                results[model_name] = {
                    'status': 'success',
                    'path': tflite_path
                }
                print(f"✓ {model_name} converted successfully")
                
            except Exception as e:
                results[model_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
                print(f"✗ {model_name} conversion failed: {e}")
        
        # Print summary
        print(f"\n{'='*60}")
        print("CONVERSION SUMMARY")
        print(f"{'='*60}")
        
        for model_name, result in results.items():
            status = result['status']
            if status == 'success':
                print(f"✓ {model_name}: SUCCESS")
            else:
                print(f"✗ {model_name}: FAILED - {result['error']}")
        
        return results


# Usage example
def main():
    """Example usage of the EdgeFace converter"""
    
    # Import your original get_model function
    # from your_edgeface_module import get_model
    
    converter = EdgeFaceConverter()
    
    # Convert single model
    # converter.convert_to_tflite('edgeface_xxs', get_model)
    
    # Convert multiple models
    models_to_convert = [
        'edgeface_xxs',           # Base model (no LoRA, no quantization)
        'edgeface_base',          # Base model  
        'edgeface_xs_gamma_06',   # LoRA model
        'edgeface_s_gamma_05',    # LoRA model
    ]
    
    # converter.batch_convert(models_to_convert, get_model)
    
    print("Import this module and use:")
    print("converter = EdgeFaceConverter()")
    print("converter.convert_to_tflite('model_name', get_model)")

if __name__ == "__main__":
    main()