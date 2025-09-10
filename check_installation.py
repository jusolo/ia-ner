#!/usr/bin/env python3
"""
Script para verificar que todas las dependencias estén instaladas correctamente
"""

import sys
import importlib.util

def check_package(package_name, import_name=None):
    """Verificar si un paquete está instalado"""
    if import_name is None:
        import_name = package_name
    
    try:
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            print(f"✅ {package_name}: Instalado")
            return True
        else:
            print(f"❌ {package_name}: NO encontrado")
            return False
    except Exception as e:
        print(f"❌ {package_name}: Error - {e}")
        return False

def main():
    """Verificar todas las dependencias"""
    print("🔍 VERIFICANDO INSTALACIÓN")
    print("=" * 40)
    
    packages = [
        ("torch", "torch"),
        ("transformers", "transformers"),
        ("datasets", "datasets"),
        ("tokenizers", "tokenizers"),
        ("accelerate", "accelerate"),
        ("evaluate", "evaluate"),
        ("seqeval", "seqeval"),
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("scikit-learn", "sklearn"),
        ("tqdm", "tqdm"),
        ("pyyaml", "yaml")
    ]
    
    all_installed = True
    for package_name, import_name in packages:
        if not check_package(package_name, import_name):
            all_installed = False
    
    print("\n" + "=" * 40)
    if all_installed:
        print("🎉 ¡Todas las dependencias están instaladas!")
        
        # Verificar versiones importantes
        print("\n📋 Versiones importantes:")
        try:
            import torch
            print(f"PyTorch: {torch.__version__}")
        except:
            pass
            
        try:
            import transformers
            print(f"Transformers: {transformers.__version__}")
        except:
            pass
            
        try:
            import datasets
            print(f"Datasets: {datasets.__version__}")
        except:
            pass
        
        # Verificar CUDA
        try:
            import torch
            if torch.cuda.is_available():
                print(f"🚀 CUDA disponible: {torch.cuda.get_device_name(0)}")
            else:
                print("💻 Usando CPU (CUDA no disponible)")
        except:
            pass
            
        print("\n✅ El proyecto está listo para usar!")
        
    else:
        print("❌ Algunas dependencias faltan.")
        print("\nPara instalar:")
        print("pip install -r requirements.txt")
    
    return all_installed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)