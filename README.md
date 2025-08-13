Dudas:
Batchnorm2d? Se integra con la capa quantizada (folded) durante la inferencia? Hailo lo hace pero aqui que pasa?

Pasos:
1.
Meter VGGT en el pipeline aunque sea para poder hacer la inferencia de imagenet con el checkpoint comercial
Registrar de otra forma los parametros en vez de register_buffer porque solo se queda el ultimo batch y es memory intensive para la gpu. Hacerlo como una lista self.saved_outpus = {"x": [], "yq": [], ...}. Probar module.register_forward_hook
Empezar a mirar PTQ
2.
Descargar el dataset de 5TB
3.
Ver como hacer modelo entero int

ToDo:
Sustituir run_test de qat por STATS

Remarks:
Importance of first and last layer quantization (8 bit, the others 2-3-4 bit)