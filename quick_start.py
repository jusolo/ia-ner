#!/usr/bin/env python3
"""
Script de inicio rápido para el proyecto NER
Ejecuta todo el pipeline: descarga -> entrenamiento -> evaluación
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(command, description):
    """Ejecutar comando y mostrar resultado"""
    print(f"\n{'='*50}")
    print(f"EJECUTANDO: {description}")
    print(f"Comando: {command}")
    print('='*50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=True, text=True)
        print("✅ ÉXITO")
        if result.stdout:
            print("Salida:")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print("❌ ERROR")
        print(f"Código de error: {e.returncode}")
        if e.stderr:
            print("Error:")
            print(e.stderr)
        return False

def main():
    """Función principal"""
    print("🚀 INICIO RÁPIDO - ENTRENAMIENTO NER")
    print("Este script ejecutará todo el pipeline automaticamente")
    
    # Verificar que estamos en el directorio correcto
    if not os.path.exists('config.yaml'):
        print("❌ Error: No se encontró config.yaml")
        print("Asegúrate de estar en el directorio del proyecto")
        sys.exit(1)
    
    # Paso 0: Verificar instalación
    print("\n🔍 Verificando instalación...")
    if not run_command("python check_installation.py", "Verificando dependencias"):
        print("❌ Hay problemas con las dependencias. Ejecuta 'make install' primero.")
        sys.exit(1)
    
    # Paso 1: Descargar datos
    if not run_command("python src/download_data.py", "Descargando dataset"):
        print("❌ Error descargando datos. Abortando.")
        sys.exit(1)
    
    # Paso 2: Entrenar modelo
    if not run_command("python src/train.py", "Entrenando modelo NER"):
        print("❌ Error entrenando modelo. Abortando.")
        sys.exit(1)
    
    # Paso 3: Evaluar modelo  
    if not run_command("python src/evaluate.py", "Evaluando modelo"):
        print("❌ Error evaluando modelo.")
        # No salimos aquí porque el entrenamiento fue exitoso
    
    print(f"\n{'='*50}")
    print("🎉 ¡PIPELINE COMPLETADO!")
    print("='*50")
    print("Resultados:")
    print(f"📁 Modelo guardado en: ./models/")
    print(f"📁 Datos en: ./data/")
    print("\nPróximos pasos:")
    print("- Revisa los resultados de evaluación arriba")
    print("- Modifica config.yaml para experimentar")  
    print("- Usa 'python src/test_model.py' para probar con nuevo texto")
    print("- Ejecuta 'make help' para ver más comandos")

if __name__ == "__main__":
    main()