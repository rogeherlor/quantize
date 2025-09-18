Dudas:
Batchnorm2d? Se integra con la capa quantizada (folded) durante la inferencia? Hailo lo hace pero aqui que pasa?

Pasos:


Entrenar:
    Pasar GPTQ a LSQ
    Entrenar

PTQ:
    Comprobar GPTQ y ver que funciona igual la transferencia a nuestra clasee LSQ_Quantize
    Mirar Outlier Suppression / Outlier Suppression+ (Park et al., 2023)
    GPTQ + SmoothQuant is the state of the art?

Distill:
    ZeroQuant-V2 y original

Comprobar pasar a nuestro LSQ_quantizer formato funciona bien. Estudiar si penaliza mucho que siempre sea symmetric
Intentar fine-tuning
Ver si fine tune a la vez o hacer lo de capa por capa viendo alguna metrica para saber cual hacer

ToDo:
Sustituir run_test de qat por STATS
Ver si se puede optimizar algo en memoria, pero de moemnto no. Si se quiere correr el original con batch 48 y accum steps 2 con quantizavion probablemente hace falta GPU 80GB
Seria bueno tener un scale por cada channel en conv, pero igual HAILO no lo soporta

Remarks:
Importance of first and last layer quantization (8 bit, the others 2-3-4 bit)