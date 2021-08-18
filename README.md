# DientesMask

Basados en los últimos avances en segmentación semántica de dientes en radiografías panorámicas, se propone el siguiente flujo de trabajo:

1. Segmentación semántica usando una red MaskRCNN.

2. Corrección de las etiquetas producidas.

Actualmente, la red MaskRCNN identifica y clasifica las instancias detectadas entre Molar, Premolar, Canino, Incisivo, y posteriormente un algoritmo registra las instancias con su correspondiente numeración. Sin embargo, sería posible cambiar las clases producidas por la red y el algoritmo de detección, si es que se requiere, y mantener el flujo de trabajo propuesto.

Las imagenes fueron segmentadas con https://github.com/wkentaro/labelme .

El repo no tiene los datos. En el notebook de entrenamiento se hace referencia a los directorios  
- data/images  
- data/jsons/train  
- data/jsons/test  
- 
que son el directorio con las imagenes, el directorio con los jsons de las imagenes de entrenamiento, y el directorio con los jsons de las imagenes de test 

Los scripts principales son: 

**Training_mask_rcnn.ipynb:** Notebook para entrenar la red neuronal.

**Demo_mask_rcnn_predictions_corrections.ipynb:** Demo de red+corrección de etiquetas.

**Automatic_annotation_of_new_images.ipynb:** Notebook para segmentar automaticamente nuevas imagenes y agrandar dataset. Útil para facilitar el etiquetado manual, las imagenes producidas por el notebook necesitan revision!

**utils.py:** Por ahora el único script con funciones útiles. La idea es hacer código modular y prolijo.

El entrenamiento se hace con el framework Detectron2 de facebook. Además, los notebooks usan Google Drive principalmente para guardar los modelos durante el entrenamiento.
