#imports
import numpy as np
import cv2
from scipy import optimize,interpolate
from google.colab.patches import cv2_imshow
import torch
import os
import json
import base64

from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

#load a mask rcnn
def make_maskrcnn_predictor(model_path, num_classes, prediction_threshold):
  cfg = get_cfg()
  cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
  cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = prediction_threshold # set threshold for this model
  cfg.MODEL.WEIGHTS = model_path
  cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes
  predictor = DefaultPredictor(cfg)
  return predictor

def open_image(img_path):
  return cv2.imread(img_path)

#predict an image using a mask rcnn
def predict_image(img, predictor):
  #prediccion
  outputs = predictor(img)
  predictions = outputs["instances"].to("cpu")
  #remover duplicados de predictions:
  masks = predictions.pred_masks
  scores = predictions.scores
  masks_len = len(masks)
  index_list = list(range(masks_len))
  th = 0.7
  overlapping_preds = []
  for i in range(masks_len-1):
    for j in range(i+1,masks_len):
      overlap_count = (masks[i] * masks[j]).sum().item()
      if (overlap_count > masks[i].sum().item()*th) or (overlap_count > masks[j].sum().item()*th):
        overlapping_preds.append((i,j))
  for (i,j) in overlapping_preds:
    index_list.remove(i) if scores[i] < scores[j] else index_list.remove(j)
  return predictions[index_list]

#visualize the results
def visualize_predictions(img, predictions, metadata):
  visualizer = Visualizer(img[:, :, ::-1],metadata = metadata, scale=2)
  vis = visualizer.draw_instance_predictions(predictions)
  cv2_imshow(vis.get_image()[:, :, ::-1])

#template matching based in pseudo cross correlation
def template_match(a,b):
  len_a, len_b = len(a), len(b)
  len_diffs = len_b - len_a
  num_aciertos = []
  for i in range(len_diffs + 1): #el bucle debe correr al menos una vez (caso len(b) == len(a))
    num_aciertos.append(0)
    b_aux = b[i:i+len_a]
    aciertos = [x == y for x,y in zip(a,b_aux) ]
    num_aciertos[-1] += sum(aciertos)
  res = np.argmax(num_aciertos)
  return res

#correct the labels of the predictions of maskrcnn
def correct_labels(predictions):
  #se separan los dientes superiores e inferiores con una regresion
  def regression_func(x,a,b,c,d):
    return a*x*x*x + b*x*x + c*x + d
  pred_boxes = predictions.pred_boxes.tensor.cpu().numpy()
  pred_centers = np.array(list(map(lambda x: [(x[0]+x[2])/2,(x[1]+x[3])/2 ] ,pred_boxes)))
  params, params_cov = optimize.curve_fit(regression_func,pred_centers[:,0], pred_centers[:,1])

  upper_teeth = [np.where(pred_centers == x)[0][0] for x in pred_centers if x[1] < regression_func(x[0], params[0],params[1],params[2],params[3])]
  lower_teeth = [np.where(pred_centers == x)[0][0] for x in pred_centers if x[1] > regression_func(x[0], params[0],params[1],params[2],params[3])]

  #ordenamiento izquierda a derecha
  upper_teeth.sort(key = lambda x :pred_centers[x,0])
  lower_teeth.sort(key = lambda x :pred_centers[x,0])

  #lista con las etiquetas correspondientes a los indices
  upper_teeth_labels = [predictions.pred_classes[i].item() for i in upper_teeth]
  lower_teeth_labels = [predictions.pred_classes[i].item() for i in lower_teeth]

  #lista con etiquetas en dentadura perfecta:
  teeth_order = [0,0,0,1,1,2,3,3,3,3,2,1,1,0,0,0]
  #listas con las clases de los dientes enumerados, de izquierda a derecha (de 0 a 31)
  full_upper_teeth_numbers = list(range(16))
  full_lower_teeth_numbers = list(range(16,32))

  #deteccion de offset
  upper_teeth_offset = template_match(upper_teeth_labels, teeth_order)
  lower_teeth_offset = template_match(lower_teeth_labels, teeth_order)

  #etiquetas de dientes superiores e inferiores
  upper_teeth_numbers = list( map( lambda x: full_upper_teeth_numbers[x], list( range( upper_teeth_offset, upper_teeth_offset + len(upper_teeth_labels ) ) ) ) )
  lower_teeth_numbers = list( map( lambda x: full_lower_teeth_numbers[x], list( range( lower_teeth_offset, lower_teeth_offset + len(lower_teeth_labels ) ) ) ) )

  new_pred_classes_list = list(range(len(pred_boxes)))

  for i in range(len(upper_teeth_numbers)):
    idx = int( upper_teeth[i] )
    cat = upper_teeth_numbers[i]
    new_pred_classes_list[idx] = cat

  for i in range(len(lower_teeth_numbers)):
    idx = int( lower_teeth[i] )
    cat = lower_teeth_numbers[i]
    new_pred_classes_list[idx] = cat

  aux = torch.tensor(new_pred_classes_list)
  predictions.set("pred_classes", aux)
  return predictions

#function to turn binary mask to list of points of a polygon
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
        lista.append([ int(res[i]), int(res[i+1]) ])
        i+=2
    # lista = [np.array(lista)]
    return lista, has_holes

#function to make a .json out of a single image
def predict_img_and_make_json(img_path, predictor):
  #img file name without dirname
  img_name = os.path.basename(img_path)
  #make base64 of image (its a field in the .json)
  encoded = base64.b64encode(open(img_path, "rb").read())
  #predict
  img = open_image(img_path)
  inst = predict_image(img, predictor)
  #correct labels
  inst = correct_labels(inst)
  #size
  height, width = inst.image_size
  #the fields we are gonna use from the Instances object
  inst_fields = inst.get_fields()
  pred_masks = inst_fields['pred_masks']
  pred_classes = inst_fields['pred_classes']
  #empty list to append the detected instances
  shapes_list = []
  #top of 20 points per polygon
  points_amount = 20 
  #string labels for the classes
  teeth_numbers = ["18","17","16","15","14","13","12","11","21","22","23","24","25","26","27","28",\
                 "48","47","46","45","44","43","42","41","31","32","33","34","35","36","37","38"]
  #for every detected instance
  for i in range(len(pred_masks)):
    #convert the binary mask to a polygon (list of points of the polygon)
    points_list, aux = mask_to_polygons( pred_masks[i] )
    #empty list
    new_points_list = []
    #we are gonna have max 20 points per polygon
    subsampling_factor = max( len(points_list) // points_amount , 1)
    #subsample points_list
    for j in range(0, len(points_list), subsampling_factor):
      new_points_list.append( points_list[j] )
    #label
    num_label = int( pred_classes[i].cpu().numpy() ) 
    label = teeth_numbers [num_label]
    #dictionary for the polygon being processed
    shape_dict = {"line_color":None, "fill_color":None, "label":label,
                  "points":new_points_list, "group_id":None, "shape_type":"polygon","flags": {}}
    shapes_list.append(shape_dict)
  
  #make the .json
  output_dict = {
    "version": "4.2.9",
    "flags": {},
    "shapes": shapes_list,
    "imagePath":img_name,
    "imageData": str(encoded)[1::],
    "imageHeight": height,
    "imageWidth": width,
    "lineColor": [
      0,
      255,
      0,
      128
    ],
    "fillColor": [
      255,
      0,
      0,
      128
    ]
  }

  #write the .json
  output_json_file_name = img_path.split('.')[0] + '.json'
  with open(output_json_file_name, 'w') as outfile:
    json.dump(output_dict, outfile, indent=4)

#function to predict and make .jsons of all the images in a folder
def predict_folder_and_make_jsons(folder_path, predictor):
  imgs_list = os.listdir(folder_path)
  for img_file in imgs_list:
    img_path = os.path.join(folder_path, img_file)
    try:
      predict_img_and_make_json(img_path, predictor)
    except:
      continue

