# DientesMask

Basados en los últimos avances en segmentación semántica de dientes en radiografías panorámicas, se propone el siguiente flujo de trabajo:

1. Segmentación semántica usando una red MaskRCNN.

2. Corrección de las etiquetas producidas.

Actualmente, la red MaskRCNN identifica y clasifica las instancias detectadas entre Molar, Premolar, Canino, Incisivo, y posteriormente un algoritmo registra las instancias con su correspondiente numeración. Sin embargo, sería posible cambiar las clases producidas por la red y el algoritmo de detección, si es que se requiere, y mantener el flujo de trabajo propuesto.

Los scripts principales son: 

**Training_mask_rcnn.ipynb:** Notebook para entrenar la red neuronal.

**Demo_mask_rcnn_predictions_corrections.ipynb:** Demo de red+corrección de etiquetas.

**Automatic_annotation_of_new_images.ipynb:** Notebook para segmentar automaticamente nuevas imagenes y agrandar dataset.

**utils.py:** Por ahora el único script con funciones útiles. La idea es hacer código modular y prolijo.

## Flujo de trabajo con git:

El repo cuenta con dos ramas: **master** y **develop** (que es copia de master)

Para agregar/cambiar funcionalidades del repo, el desarrollador va a hacer una rama desde **develop** y trabajar ahí. Una vez el desarrollo de esa funcionalidad o cambio está finalizado, **se hará un merge desde el branch nuevo al branch develop y se eliminará la rama**.

Si todos los nuevos cambios estan bien ("autorizados" por los demás), **se hará un merge desde develop a master**.

