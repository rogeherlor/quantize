Dudas:
Batchnorm2d? Se integra con la capa quantizada (folded) durante la inferencia? Hailo lo hace pero aqui que pasa?

Pasos:
4W 8A (QAT y GPTQ)
Hardware real aceleracion
Mejorar tiempo de entrenamiento
(Bias-variance es regularizacion y no se suele aplicar en LLMs pero en quantizacion puede que aporte ¿?)
Distill, check self-supervised y temperature

Smoothquant entero
Distill

Entrenar:
    Fix resume training

PTQ:
    Intentar mejorar GPTQ sequential inference
    SmoothQuant
    GPTQ + SmoothQuant
    Mirar Outlier Suppression / Outlier Suppression+ (Park et al., 2023)

Distill:
    ZeroQuant-V2 y original


Intentar fine-tuning
Ver si fine tune a la vez o hacer lo de capa por capa viendo alguna metrica para saber cual hacer
Estudiar si penaliza que siempre sea symmetric

ToDo:
Sustituir run_test de qat por STATS
Ver si se puede optimizar algo en memoria, pero de moemnto no. Si se quiere correr el original con batch 48 y accum steps 2 con quantizavion probablemente hace falta GPU 80GB
Seria bueno tener un scale por cada channel en conv, pero igual HAILO no lo soporta

Remarks:
Importance of first and last layer quantization (8 bit, the others 2-3-4 bit)