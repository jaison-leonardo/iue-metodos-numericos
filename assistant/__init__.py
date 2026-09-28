"""
Paquete `assistant`
====================
Capa NUEVA de automatización que se agrega ALREDEDOR de los módulos
numéricos existentes (Met_busqraices.py, met_solpvi_1ord.py, etc.),
según lo definido en "Arquitectura y Especificación Técnica AMN.md"
(sección 14). No reemplaza ni modifica los algoritmos numéricos.

Principio: el LLM interpreta y explica; SymPy/SciPy validan y
resuelven simbólicamente; los métodos numéricos existentes son la
única fuente de verdad del cálculo numérico.
"""
