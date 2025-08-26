Dudas:
Batchnorm2d? Se integra con la capa quantizada (folded) durante la inferencia? Hailo lo hace pero aqui que pasa?

Pasos:
1.
Empezar a mirar PTQ
Registrar de otra forma los parametros en vez de register_buffer porque solo se queda el ultimo batch y es memory intensive para la gpu. Hacerlo como una lista self.saved_outpus = {"x": [], "yq": [], ...}. Probar module.register_forward_hook
3.
Ver como hacer modelo entero int

ToDo:
Sustituir run_test de qat por STATS
Ver si se puede optimizar algo en memoria, pero de moemnto no. Si se quiere correr el original con batch 48 y accum steps 2 con quantizavion probablemente hace falta GPU 80GB

Remarks:
Importance of first and last layer quantization (8 bit, the others 2-3-4 bit)