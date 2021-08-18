#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 03:53:25 2020

@author: snoopi
"""

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


import os
from os import listdir
from os.path import isfile, join

import numpy as np
import json
from detectron2.structures import BoxMode

from detectron2.data import DatasetCatalog, MetadataCatalog

from detectron2.engine import DefaultTrainer
from detectron2.config import get_cfg

import base64


def probar(d):
    for x in ["train", "val"]:
        try:
            DatasetCatalog.register("dientes_" + x, lambda x=x: get_dientes_dicts("Etiquetado final/" + x))
            MetadataCatalog.get("dientes_" + x).set(thing_classes=["molares","premolares","caninos","incisivos"])
        except:
            1==1
    dientes_metadata = MetadataCatalog.get("dientes_train")


    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
    cfg.DATASETS.TRAIN = ("dientes_train",)
    cfg.DATASETS.TEST = ()
    cfg.DATALOADER.NUM_WORKERS = 2
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")  # Let training initialize from model zoo

    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4  #  Clases molares, premolares, caninos y incisivos

# os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=False)


    cfg.MODEL.WEIGHTS = os.path.join("/home/snoopi/Escritorio/modeloEntrenado/model_final.pth")
    print(cfg.MODEL.WEIGHTS)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.6   # set the testing threshold for this model
    cfg.DATASETS.TEST = ("dientes_val", )
    predictor = DefaultPredictor(cfg)

    from detectron2.utils.visualizer import ColorMode


    from os import listdir
    from os.path import isfile, join

    
    im = cv2.imread(d)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                  metadata=dientes_metadata, 
                  scale=1, 
                  instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
  )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    imagen= cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_RGB2BGR)

    predictions = outputs["instances"].to("cpu")

    
    return predictions    

def mask_to_polygons(mask):
    # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
    # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
    # Internal contours (holes) are placed in hierarchy-2.
    # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
    mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
    res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    hierarchy = res[-1]
    if hierarchy is None:  # empty mask
        return [], False
    has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
    res = res[-2]
    res = [x.flatten() for x in res]
    res = [x for x in res if len(x) >= 6]
    res=res[0]
    lista = []
    i=0
    while i!=len(res):
        lista.append([res[i],res[i+1]])
        i+=2
    # lista = [np.array(lista)]
    return lista, has_holes

def puntosMedios(predictions,imagen=None):
    boxes = predictions.pred_boxes
    boxes = boxes.tensor.numpy()
    puntos_medio = [( (bx[2]+bx[0])/2, (bx[3]+bx[1])/2  ) for bx in boxes]
    
    if imagen!=None:  ##Si se le da una imagen le imprime los puntos medios 
        for i in puntos_medio:
            cv2.circle(imagen,(int(i[0]), int(i[1])), 10, (0,255,0), -1)
    
    return puntos_medio



def crear_lista_dientes(predictions):
    
    masks = np.asarray(predictions.pred_masks)
    classes = np.asarray(predictions.pred_classes)
    scores = predictions.scores if predictions.has("scores") else None
    scores=np.asarray(scores)
    puntos_medios=puntosMedios(predictions)

    masks_aux = []
    i=0
    for mask in masks:
        masks_aux.append([np.copy(mask),i])
        i+=1
    del i

    mismo_diente = []


################################################
###Se comparan las máscaras para identificar ###
###cuando un mismo diente fue reconocido como###
###de dos tipos diferentes. Se comparan las  ###
###areas y se ven si conciden en un % o más  ###
################################################
    while len(masks_aux) != 1:
        mask=masks_aux.pop()
        for mask2 in masks_aux:
            masksAND = np.logical_and(mask[0],mask2[0])  ## El AND entre las 2 mascaras
            trues = np.count_nonzero(masksAND == True)  #cuanta cuantas veces hay un valor True en la matris
            if trues != 0:
                print("se tocan ", mask[1] , " y ",mask2[1], " - veces: ",trues)
                porcentaje=0.7 #porcentaje de solapamiento entre dientes
                if (trues > (np.count_nonzero(mask[0] == True)*porcentaje)
                 or trues > (np.count_nonzero(mask2[0] == True)*porcentaje)): 
                    print("solapados")
                    mismo_diente.append((mask[1],mask2[1]))
    del masks_aux,mask,mask2,masksAND,trues
#+++++++++++++++++++++++++++++++++++++++++++++++

################################################
###Comprueba que un elementeo de masks solo  ###
###se encuentro en 1 solo par, es decir solo ### 
###se concidera el caso que un diente real se### 
###reconoció como 2 tipos de dientes diferentes#
################################################

    lista_repetidos = [x for i in mismo_diente for x in i] #se colocan el arreglo de pares en un solo arreglo
    no_se_repite=np.array([lista_repetidos.count(i) for i in lista_repetidos]) #se arma un arreglo que dice cuanta veces aparece un valor
    no_se_repite =np.where(no_se_repite==1,True, False)   
    no_se_repite=not False in no_se_repite

### Si no_se_repite es TRUE es porque los diente 
###solo se encuentra solo una vez solo esta en 1 par
###+++++++++++++++++++++++++++++++++++++++++++++

    lista_dientes = []

    if no_se_repite:
        for x in range(0,len(masks)):
            if not x in lista_repetidos:  # Si es diente que solo se detecto una vez
                mask=masks[x]
                type_score={"type":classes[x],
                            "score":scores[x]}
            
                diente =	{"mask": mask,
                            "type_score": type_score,
                            "punto_medio": puntos_medios[x]}
                lista_dientes.append(diente)                  
    
        for y,x in mismo_diente:
            mask=np.logical_or(masks[x],masks[x])  ## como máscara final me quedo (arbitrariamente) con la suma de las 2 mascaras
            list_aux=[]
            
            scores_aux=(scores[y],scores[x])
            classes_aux=(classes[y],classes[x])         ##Esto es para poner al de maypr 
            mini=scores_aux.index(min(scores_aux))      ##porcentaje primero
            maxi=scores_aux.index(max(scores_aux))
            
            list_aux.append({"type":classes_aux[maxi],
                             "score":scores_aux[maxi]})  
            list_aux.append({"type":classes_aux[mini],
                             "score":scores_aux[mini]}) 
        
            puntoMedio_promedio=(np.array(puntos_medios[x])+np.array(puntos_medios[y]))/2
            puntoMedio_promedio=list(puntoMedio_promedio)
        
            diente =	{"mask": mask,
                         "type_score": list_aux,
                         "punto_medio": puntoMedio_promedio}
        
            lista_dientes.append(diente)
    return lista_dientes









# dataset_dicts = [f for f in listdir(mypath) if isfile(join(mypath, f)) and (".jpg" in f or ".png" in f)]
def get_data(filename,folder):
    
    path=folder+"/"+filename
    predictions = probar(path)
    # filename="TOPA_MARIA_16112018_155935.jpg"
    size    = os.stat(path).st_size


    thing_classes=["molares","premolares","caninos","incisivos"]
    lista_dientes=crear_lista_dientes(predictions)

    regions = []
    for diente in lista_dientes:
        mask=diente['mask']
        clase= diente["type_score"]["type"] if not (type(diente["type_score"]) == list) else diente['type_score'][0]['type']   

    
        xy,aux=mask_to_polygons(mask)
        x_aux = [x[0] for x in xy]
        y_aux = [y[1] for y in xy]
    
        x=[]
        y=[]
        
        cantidad_puntos=20       
        cantidad_puntos_aux=int(len(x_aux)/cantidad_puntos)
        if cantidad_puntos_aux==0:
            cantidad_puntos_aux=1
            
        for i in range(0,len(x_aux),cantidad_puntos_aux):
            x.append(int(x_aux[i]))
            y.append(int(y_aux[i]))

        shape_attributes =	{"name": "polygon",
                            "all_points_x": x,
                            "all_points_y": y}
    
        clase=thing_classes[clase]
        region_attributes=  {"dientes": {clase: True}}
    
        attributes =	{"region_attributes": region_attributes,
                         "shape_attributes": shape_attributes}
    
        regions.append(attributes)

    fotoEtiquetada = {"file_attributes": {},
                      "filename": filename,
                      "regions":regions,
                      "size":   size}
    return fotoEtiquetada


def autoEtiquetado(folder,guardarEn=None,unirA=None):
    from os import path
    dataset = [f for f in listdir(folder) if isfile(join(folder, f)) and (".jpg" in f or ".png" in f)]
    
    if len(dataset)==0:
        return
    
    if unirA!=None:
      if not path.exists(unirA):
          print(unirA," NO existe")
          return
    
    primera_imagen=dataset.pop()
    print(primera_imagen)
    path=folder+"/"+primera_imagen
    size    = os.stat(path).st_size
    data=get_data(primera_imagen,folder)
    
    nombre=primera_imagen+str(size)
    etiquetado = {nombre:data }
            
    for nombre_imagen in dataset:
        
        print(nombre_imagen)
        path=folder+"/"+nombre_imagen
        size    = os.stat(path).st_size
        data=get_data(nombre_imagen,folder)
    
        nombre=nombre_imagen+str(size)
        etiquetado.update({nombre:data })
        
    if unirA!=None:
        with open(unirA) as f:
            unir_A = json.load(f)
        etiquetado.update(unir_A)

    if guardarEn!=None:
        with open(guardarEn+'.json', 'w') as json_file:
            json.dump(etiquetado, json_file)

    return etiquetado        



def unir_2_json(json1,json2,nombre_salida): ##path de los archivos .son

    with open(json1) as f:
        json_1 = json.load(f)
    
    with open(json2) as f:
        json_2 = json.load(f)

    json_1.update(json_2)
    
    with open(nombre_salida+'.json', 'w+') as json_file:
         json.dump(json_1, json_file)





###################################################                                             
###              FUNCIONES PARA                 ###
###                 LABELME                     ###
###################################################



def get_data_labelme(filename,folder,polygon=False):

    path=folder+"/"+filename

    thing_classes=["molares","premolares","caninos","incisivos"]

 
    with open(path, "rb") as imageFile:
        imageData = base64.b64encode(imageFile.read())
        imageData = str(imageData)
        imageData=imageData[1::]
    
    predictions = probar(path)

    imageHeight,imageWidth = predictions.image_size

    # thing_classes=["molares","premolares","caninos","incisivos"]
    
    lista_dientes=crear_lista_dientes(predictions)
    
    shape_type="polygon" if polygon else "rectangle"
    
    shapes = []
    i=1

    for diente in lista_dientes:
        points = []
        mask=diente['mask']
        clase= diente["type_score"]["type"] if not (type(diente["type_score"]) == list) else diente['type_score'][0]['type']   
        
        xy,aux=mask_to_polygons(mask)
        point = []
        x_aux = [float(x[0]) for x in xy]
        y_aux = [float(y[1]) for y in xy]
        if not polygon:
            xmin=(min(x_aux))
            xmax=(max(x_aux))
            ymin=(min(y_aux))
            ymax=(max(y_aux))
            points=[[xmin,ymin],[xmax,ymax]]
            label=str(i)
        else:
            label=thing_classes[clase]
            cantidad_puntos=20       
            cantidad_puntos_aux=int(len(x_aux)/cantidad_puntos)
            if cantidad_puntos_aux==0:
                cantidad_puntos_aux=1
            
            for i in range(0,len(x_aux),cantidad_puntos_aux):
                points.append([x_aux[i],y_aux[i]])
            
            
        shape = {"line_color": None,
                 "fill_color": None,
                 "label"     : label,
                 "points"    : points,
                 "group_id"  : None,
                 "shape_type": shape_type,
                 "flags"     : {}}
                 

        
        i+=1
        
        shapes.append(shape)
        
        
    fotoEtiquetada = {'version'     : "4.2.9",
                      'flags'       : {},
                      'shapes'      : shapes,
                      'imagePath'   : filename,
                      'imageData'   : imageData,
                      'imageHeight' : imageHeight,
                      'imageWidth'  : imageWidth,
                      'lineColor'   : [0, 255, 0, 128],
                      'fillColor'   : [255, 0, 0, 128]}
    
    return fotoEtiquetada

    
        
    


def autoEtiquetado_labelme(folder,polygon=False):
    dataset = [f for f in listdir(folder) if isfile(join(folder, f)) and (".jpg" in f or ".png" in f)]
    
    
    for nombre_imagen in dataset:
        print(nombre_imagen)
        data=get_data_labelme(nombre_imagen,folder,polygon)
        with open(folder+"/"+nombre_imagen[0:-4]+'.json', 'w+') as json_file:
            json.dump(data, json_file)
            
            
            
def jsonVia_to_labelme(jsonVia_name,folder=''):
    thing_classes=["molares","premolares","caninos","incisivos"]
    name = jsonVia_name if jsonVia_name.count(".json") else (jsonVia_name+".json")
    
    with open(folder+"/"+name) as f:
        jsonVia = json.load(f)

    
    for key in jsonVia.keys():
        aux=jsonVia[key]
        
        regions=aux['regions']
        shapes = []
        print(key)
        for region in regions:
            shape_attributes = region['shape_attributes']
            region_attributes= region['region_attributes']['dientes']
            filename = aux['filename']
        
            if type(region_attributes)==str:
                label=region_attributes
            else:
                for clase in thing_classes:
                    label=clase
                    if clase in region_attributes.keys():
                        break
                
                
            x=shape_attributes['all_points_x']
            y=shape_attributes['all_points_y']
            points=[]
            for i in range(0,len(x)):
                points.append( [ float(x[i]) , float(y[i]) ] )

            shape = {"line_color": None,
                     "fill_color": None,
                     "label"     : label,
                     "points"    : points,
                     "group_id"  : None,
                     "shape_type": 'polygon',
                     "flags"     : {}}
            
            shapes.append(shape)
            with open(folder+'/'+filename, "rb") as imageFile:
                imageData = base64.b64encode(imageFile.read())
                imageData = str(imageData)
                imageData=imageData[1::]
            
        im = cv2.imread(folder+'/'+filename)
        imageHeight, imageWidth, channels = im.shape
        
        fotoEtiquetada = {'version'     : "4.2.9",
                          'flags'       : {},
                          'shapes'      : shapes,
                          'imagePath'   : filename,
                          'imageData'   : imageData,
                          'imageHeight' : imageHeight,
                          'imageWidth'  : imageWidth,
                          'lineColor'   : [0, 255, 0, 128],
                          'fillColor'   : [255, 0, 0, 128]}

        with open(folder+'/'+filename[0:-4]+'.json', 'w+') as json_file:
            json.dump(fotoEtiquetada, json_file)
