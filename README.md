Dudas:
Batchnorm2d? Se integra con la capa quantizada (folded) durante la inferencia? Hailo lo hace pero aqui que pasa?

Pasos:
Obtener sus resultados del paper (train)
Añadir quantizacion attention para ViT
Añadir quantizacion Batch2D
Ver como hacer modelo entero int

Todo:
Registrar de otra forma los parametros en vez de register_buffer porque solo se queda el ultimo batch y es memory intensive para la gpu. Hacerlo como una lista self.saved_outpus = {"x": [], "yq": [], ...}
Sustituir run_test de qat por STATS
Importance of first and last layer quantization (8 bit, the others 2-3-4 bit)