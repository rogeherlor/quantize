Dudas:
Batchnorm2d? Se integra con la capa quantizada (folded) durante la inferencia? Hailo lo hace pero aqui que pasa?

Pasos:
Cuando se carga el modelo no se pueden guardar los pesos del pth porque el modelo no inicializa x_scale y w_scale
multiplicar por x_scale para obtener int en stats
Obtener sus resultados del paper (train, ver tipo y valor de los pesos quantizados, ver histograma, ver activaciones en tensorboard)
    Entrenar quantizados
    Visualizar pesos quantizados
    Visualizar activaciones quantizadas
    Visualizar vit
Añadir quantizacion Batch2D
Añadir ViT
Añadir quantizacion attention
Igual modulo dequant para poder obtener mas facilmente los valores quantizados