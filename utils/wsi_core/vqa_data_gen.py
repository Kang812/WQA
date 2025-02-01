import sys
sys.path.append("/workspace/whole_slide_image_LLM/wsi_level_vqa-main/")
import albumentations as A
import numpy as np
import pandas as pd
import os
import cv2
import torch
import random
from utils.HIPT.hipt_4k import HIPT_4K
from utils.HIPT.hipt_model_utils import get_vit256, get_vit4k, eval_transforms
from utils.HIPT.hipt_heatmap_utils import *
from PIL import Image
from tqdm import tqdm
from mmseg.apis import init_model, inference_model


def wsi_representation_features(seg_model, HIPT_model, img):

    transforms = A.Compose([
            A.LongestMaxSize(max_size=4096),
            A.PadIfNeeded(
                min_height=4096,
                min_width=4096,
                border_mode=cv2.BORDER_CONSTANT,
                value=(255, 255, 255)  # RGB 흰색
            ),
        ])
    
    result = inference_model(seg_model, img)
    pred_mask = result.pred_sem_seg.data.detach().cpu().numpy()[0]
    pred_mask = pred_mask.astype(np.uint8)

    contours, _ = cv2.findContours(pred_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 500:
            polygon = np.ravel(contour)
            polygon_x = [polygon[i] for i in range(len(polygon)) if i % 2 == 0]
            polygon_y = [polygon[i] for i in range(len(polygon)) if i % 2 == 1]

            xmin = min(polygon_x)
            ymin = min(polygon_y)
            xmax = max(polygon_x)
            ymax = max(polygon_y)
            bboxes.append([xmin, ymin, xmax, ymax])

    features = []
    for bbox in bboxes:
        crop_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2], :]
        transform_img = transforms(image = crop_img)['image']
        region = Image.fromarray(transform_img)
        x = eval_transforms()(region).unsqueeze(dim=0)
        features.append(HIPT_model.forward(x))
    
    features = torch.vstack(features)
    return features

def wsi_diagnosis_all(data, mode = 'train'):
    
    if random.random() > 0.5:
        q = '해당 영상을 진단해줘'
    else:
        q = '해당 wsi 영상을 진단해줘'
    
    text = ''
    nan_check = str(data['진단명'].values[0])
    if nan_check != 'nan':
        if data['진단명'].values[0] == 1:
            text += "현재 Ductal(관상, 유관)으로 보입니다."
        elif data['진단명'].values[0] == 2:
            text += "현재 Lobular(소엽, 소엽상)으로 보입니다."
        elif data['진단명'].values[0] == 3:
            text += "현재 Mucinous(점액성)으로 보입니다."
        elif data['진단명'].values[0] == 4:
            pass
    else:
        pass

    nan_check = str(data['NG'].values[0])
    if nan_check != 'nan':
        if int(data['NG'].values[0]) == 1:
            text += " 현재 whole slide image의 NG는 NG1입니다. 세포의 핵이 정상에 가깝게 보이며, 크기와 모양이 균일합니다. 분열(유사분열, mitosis)이 적고, 성장이 느린 암으로 간주됩니다. 예후가 상대적으로 좋습니다."
        elif int(data['NG'].values[0]) == 2:
            text += " 현재 whole slide image의 NG는 NG2입니다. 핵이 약간 비정상적이며, 크기와 모양의 다양성이 있으며, 분열이 NG1보다 더 활발합니다. 중간 정도의 성장 속도를 가진 암으로 평가되고, 예후는 NG1보다 다소 나쁘지만 NG3보다는 낫습니다."
        elif int(data['NG'].values[0]) == 3:
            text += " 현재 whole slide image의 NG는 NG3입니다. 핵이 매우 비정상적으로 보이며, 크기와 모양의 불규칙성이 심합니다. 분열이 매우 활발하며, 빠르게 성장하고 전이 가능성이 높은 암입니다. 예후가 가장 나쁩니다."
    else:
        pass

    nan_check = str(data['암의 위치'].values[0])
    if nan_check != 'nan':
        if int(data['암의 위치'].values[0]) == 1:
            text += " 암의 위치는 오른쪽 조직 있는 것으로 보입니다."
        elif int(data['암의 위치'].values[0]) == 2:
            text += " 암의 위치는 왼쪽에 조직에 있는 것으로 보입니다."
        elif int(data['암의 위치'].values[0]) == 3:
            text += " 암은 양쪽 조직에 있는 것으로 보입니다."
    else:
        pass
    
    nan_check = str(data['암의 개수'].values[0])
    if nan_check != 'nan':
        if int(data['암의 개수'].values[0]) == 1:
            text += '암의 개수는 1개인 것으로 보이며,'+ f"가장 큰 암의 정경은 {data['암의 장경'].values[0]}mm 인것으로 보입니다."
        elif int(data['암의 개수'].values[0]) == 2:
            text += '암의 개수는 1개 이상인 것으로 보이며,'+ f"가장 큰 암의 정경은 {data['암의 장경'].values[0]}mm 인것으로 보입니다."
    else:
        pass

    nan_check = str(data['HG'].values[0])
    if nan_check != 'nan':
        if int(data['HG'].values[0]) == 1:
            text += " 현재 HG의 등급은 Grade 1 이며, 세포가 정상에 가까습니다. 또한, 낮은 악성도, 성장 및 전이 속도는 느린측에 속합니다."
        elif int(data['HG'].values[0]) == 2:
            text += " 현재 HG의 등급은 Grade 2이며, 중간 정도의 비정상성을 가지고 있으며, 중간 악성도를 가지고 있습니다."
        elif int(data['HG'].values[0]) == 3:
            text += " 현재 HG의 등급은 Grade 3이며, 세포가 매우 비정상적이며, 높은 악성도, 성장 및 전이 속도 빠름니다."
        elif int(data['HG'].values[0]) == 4:
            text += " 미세침습만 존재하는 상황입니다. 전이 가능성은 낮지만 면밀한 관찰이 필요해보입니다."
    else:
        pass

    nan_check = str(data['DCIS_or_LCIS_여부'].values[0])
    if nan_check != 'nan':
        if data['DCIS_or_LCIS_여부'].values[0] == 0:
            text += " 조직 내에서 DCIS나 LCIS가 관찰되지 않았습니다."
        elif data['DCIS_or_LCIS_여부'].values[0] == 1:
            text += " 조직 내에서 DCIS 또는 LCIS가 존재하지만, EIC가 없는 상태입니다."
        elif data['DCIS_or_LCIS_여부'].values[0] == 2:
            text += " 조직 내에서 DCIS 또는 LCIS가 존재하며, EIC가 있는 상태이며, 암세포가 유관 내부에서 넓은 영역을 차지하고 있습니다."
    else:
        pass

    if mode == 'train':
        if data['N_category'].values[0] == 0:
            text += ' 현재는 암이 림프절로 전이 되지 않은 걸로 보입니다.'
        else:
            text += ' 현재는 암이 림프절로 전이 된 걸로 보입니다.'
    return q, text

def ng_qa(data, mode):
    NG_Q_list = []
    NG_A_list = []
    NG_Q = '이 조직의 NG는 무엇인가요?'

    if int(data['NG'].values[0]) == 1:
        NG_A = "현재 whole slide image의 NG는 NG1입니다. 세포의 핵이 정상에 가깝게 보이며, 크기와 모양이 균일합니다. 유사분열(분열)이 적고 성장이 느린 암으로 간주됩니다. 예후가 상대적으로 좋습니다."
    elif int(data['NG'].values[0]) == 2:
        NG_A = "현재 whole slide image의 NG는 NG2입니다. 핵이 약간 비정상적이며, 크기와 모양의 다양성이 있습니다. 분열이 NG1보다 더 활발하며, 중간 정도의 성장 속도를 가진 암으로 평가됩니다. 예후는 NG1보다 나쁘지만 NG3보다는 낫습니다."
    elif int(data['NG'].values[0]) == 3:
        NG_A = "현재 whole slide image의 NG는 NG3입니다. 핵이 매우 비정상적으로 보이고, 크기와 모양의 불규칙성이 심합니다. 분열이 매우 활발하며 빠르게 성장하고 전이 가능성이 높은 암입니다. 예후가 가장 나쁩니다."
    NG_Q_list.append(NG_Q)
    NG_A_list.append(NG_A)

    if int(data['NG'].values[0]) == 1 and mode == 'train':
        if random.random() < 0.5:
            NG_Q = 'NG1 조직에서도 암의 위험 요소가 발견될 가능성이 있나요 ?'
        elif random.random() < 0.5:
            NG_Q = 'NG1에서의 전이 가능성이 있는지 진단해줘 ?'

        if data['N_category'].values[0] == 0:
            n_category = '현재는 암이 림프절로 전이 되지 않은 걸로 보입니다.'
        else:
            n_category = '현재는 암이 림프절로 전이 된 걸로 보입니다.'
        NG_A = f'NG1 조직은 암의 위험이 낮지만, 드물게 전이 가능성을 가진 세포군이 포함될 수 있습니다. 현재 조직의 경우, {n_category}'
        NG_Q_list.append(NG_Q)
        NG_A_list.append(NG_A)
    
    elif int(data['NG'].values[0]) == 2 and mode == 'train':
        NG_Q = 'NG2 조직에서 전이 가능성이 있나요?'
        if data['N_category'].values[0] == 0:
            n_category = '현재는 암이 림프절로 전이 되지 않은 걸로 보입니다.'
        else:
            n_category = '현재는 암이 림프절로 전이 된 걸로 보입니다.'
        
        NG_A = f'핵의 비정상성과 분열 활동성이 NG1보다 증가했기 때문에, 암세포가 성장하거나 주변 조직으로 침습할 가능성이 있습니다. 그러나 NG3처럼 전이가 매우 높은 상태는 아니며, 전이 가능성은 조직의 위치, 암세포의 밀도, 혈관이나 림프관 주변 조직 침습 여부에 따라 달라집니다. {n_category}'
        NG_Q_list.append(NG_Q)
        NG_A_list.append(NG_A)
    
    elif int(data['NG'].values[0]) == 3 and mode == 'train':
        NG_Q = 'NG3 조직에서 전이 가능성이 있나요?'
        if data['N_category'].values[0] == 0:
            n_category = '현재는 암이 림프절로 전이 되지 않은 걸로 보입니다.'
        else:
            n_category = '현재는 암이 림프절로 전이 된 걸로 보입니다.'
        
        NG_A = f'NG3 조직은 전이 가능성이 매우 높습니다. NG3는 핵의 불규칙성, 과도한 세포 분열, 높은 암세포 밀도를 특징으로 하며, 이러한 특징은 암세포의 빠른 성장과 전이를 강하게 암시합니다. 특히 혈관 침습이나 림프관 침습이 함께 관찰되는 경우, 전이 가능성이 더욱 높아집니다. {n_category}'
        NG_Q_list.append(NG_Q)
        NG_A_list.append(NG_A)

    return NG_Q_list, NG_A_list

def tumor_position_qa(data):
    Q_list = []
    A_list = []

    Q = '현재 whole slide에서의 암의 위치를 알려줘'
    if int(data['암의 위치'].values[0]) == 1:
        A = "암의 위치는 오른쪽 조직 있는 것으로 보입니다."
    elif int(data['암의 위치'].values[0]) == 2:
        A = "암의 위치는 왼쪽에 조직에 있는 것으로 보입니다."
    elif int(data['암의 위치'].values[0]) == 3:
        A = "암은 양쪽 조직에 있는 것으로 보입니다."
    Q_list.append(Q)
    A_list.append(A)
    return Q_list, A_list

def tumor_counts_qa(data):
    Q_list = []
    A_list = []
    Q = '현재 암의 개수를 알려줘'
    if int(data['암의 개수'].values[0]) == 1:
        A = '암의 개수는 1개인 것으로 보이며,'+ f"가장 큰 암의 정경은 {data['암의 장경'].values[0]}mm 인것으로 보입니다."
    elif int(data['암의 개수'].values[0]) == 2:
        A = '암의 개수는 1개 이상인 것으로 보이며,'+ f"가장 큰 암의 정경은 {data['암의 장경'].values[0]}mm 인것으로 보입니다."
    Q_list.append(Q)
    A_list.append(A)
    return Q_list, A_list

def hg_qa(data, mode):
    Q_list = []
    A_list = []
    
    if data['N_category'].values[0] == 0:
        n_category = '현재는 암이 림프절로 전이 되지 않은 걸로 보입니다.'
    else:
        n_category = '현재는 암이 림프절로 전이 된 걸로 보입니다.'

    Q = '현재 영상의 HG 등급을 진단해줘'
    if int(data['HG'].values[0]) == 1:
        A = " 현재 HG의 등급은 Grade 1 이며, 세포가 정상에 가까습니다. 또한, 낮은 악성도, 성장 및 전이 속도는 느린측에 속합니다."
    elif int(data['HG'].values[0]) == 2:
        A = " 현재 HG의 등급은 Grade 2이며, 중간 정도의 비정상성을 가지고 있으며, 중간 악성도를 가지고 있습니다."
    elif int(data['HG'].values[0]) == 3:
        A = " 현재 HG의 등급은 Grade 3이며, 세포가 매우 비정상적이며, 높은 악성도, 성장 및 전이 속도 빠름니다."
    elif int(data['HG'].values[0]) == 4:
        A = " 미세침습만 존재하는 상황입니다. 전이 가능성은 낮지만 면밀한 관찰이 필요해보입니다."
    
    Q_list.append(Q)
    A_list.append(A)

    if int(data['HG'].values[0]) == 1:
        Q = 'HG(Histologic Grade) 1 등급인 경우에도 전이 가능성이 있나요?'
        A = f"""HG 1 등급(Histologic Grade 1)은 일반적으로 전이 가능성이 낮습니다, 그러나 완전히 배제할 수는 없습니다. HG 1의 주요 특징은 조직학적 구조가 정상에 가깝고 세포의 분화도가 높고, 
        암세포가 조직 내에서 정상 세포와 비슷한 배열을 보이며, 분열 활동성이 적습니다. 암의 성장 속도가 느리고, 침습적 특성이 덜하지만, HG 1에서도 전이가 발생할 가능성이 있는 경우가 있습니다.
        HG 1이라 하더라도 혈관이나 림프관에 침습이 발견되면, 전이 가능성이 생길 수 있으며, 특정 부위(예: 혈관 밀집 지역)에 위치한 HG 1 종양은 전이 가능성을 높일 수 있습니다. 또한, 종양 크기가 크다면, 
        HG 1이라도 주변 조직에 영향을 미칠 가능성이 증가합니다. {n_category}"""
        Q_list.append(Q)
        A_list.append(A)
    
    elif int(data['HG'].values[0]) == 2:
        Q = 'HG(Histologic Grade) 2 등급인 경우에도 전이 가능성이 있나요?'
        A = f"""네, HG 2 등급(Histologic Grade 2)은 중간 정도의 전이 가능성을 가질 수 있습니다. HG2 등급의 주요한 특징은 암세포의 분화도가 중간 정도로, 정상 세포와 암세포의 구조적 차이가 명확하게 나타나기 시작하고, 세포 분열 활동이 HG 1보다 더 활발하며, 
        종양이 더 빠르게 성장할 가능성이 있습니다. HG 2에서 전이가 발생할 수 있는 이유는 암세포의 분화도가 정상보다 낮아, 침습성과 전이 가능성이 증가, 유사분열 활동이 HG 1보다 활발해 암세포가 증식할 가능성이 있고, HG 2 암은 주변 혈관이나 림프관에 침습할 
        가능성이 높아 전이가 발생할 수 있습니다. {n_category}"""
        Q_list.append(Q)
        A_list.append(A)

    elif int(data['HG'].values[0]) == 3:
        Q = 'HG(Histologic Grade) 3 등급인 경우에도 전이 가능성이 있나요?'
        A = f"""네, HG 3 등급(Histologic Grade 3)은 전이 가능성이 매우 높습니다. HG 3의 주요 특징으로는 암세포의 분화도가 매우 낮으며, 정상 조직과 암세포 간의 구조적 차이가 극명하게 나타납니다. 또한, 세포 분열 활동이 매우 활발하며, 암세포가 빠르게 증식하고 성장합니다.
        조직 경계가 불명확하여 암세포가 주변 조직으로 쉽게 침습할 가능성도 있습니다. HG 3에서 전이 가능성이 높은 이유는 HG 3은 세포가 구조적 규칙성을 잃고 주변 조직으로 쉽게 확산될 수 있습니다. 또한, 유사분열이 빈번하게 발생하며, 암세포의 증식 속도가 빠릅니다.
        HG 3 암세포는 혈관과 림프관을 통해 원격 전이를 발생시킬 가능성이 매우 높은 상태입니다. 또한, 세포의 크기와 모양이 극도로 불규칙하며, 침습적 특성이 강합니다. {n_category}"""
        Q_list.append(Q)
        A_list.append(A)

    elif int(data['HG'].values[0]) == 4:
        Q = '미세침습만 존재하는 경우에도 전이 가능성이 있나요?'
        A = f"""네, 미세침습만 존재하는 경우에도 전이 가능성이 있습니다. 그러나 그 가능성은 조직의 특성과 침습의 정도에 따라 달라집니다. 미세침습은 암세포가 기저막(Basement Membrane)을 살짝 침범하여 주변 조직으로 소량의 세포가 확산된 상태를 의미하는데, 
        일반적으로 침습 깊이가 1mm 이하로 제한됩니다. 미세침습에서 전이 가능성이 낮은 이유는 암세포가 주변 조직으로 확산된 범위가 작기 때문에 전이가 발생할 확률이 낮고, 미세침습 상태에서는 혈관이나 림프관에 침투할 가능성이 상대적으로 낮기 때문입니다.
        미세침습에서 전이가 발생할 수 있는 경우는 미세침습이 혈관이나 림프관과 가까운 위치에서 발생한 경우, 암세포가 혈행성 또는 림프성 전이를 일으킬 가능성이 있고, 암세포가 높은 증식 속도(KI-67 증가)나 강한 침습성을 보이는 경우, 미세침습의 경계가 불명확하고 
        주변 조직으로 확산된 암세포가 존재할 경우, 전이가 발생할 가능성이 있습니다. {n_category}"""
        Q_list.append(Q)
        A_list.append(A)

    return Q_list, A_list

def dcis_or_lcis(data):
    Q_list = []
    A_list = []
    
    if data['N_category'].values[0] == 0:
        n_category = '현재는 암이 림프절로 전이 되지 않은 걸로 보입니다.'
    else:
        n_category = '현재는 암이 림프절로 전이 된 걸로 보입니다.'
    
    Q = '현재 영상에서 DCIS 또는 LCIS가 있는지 진단해줘'
    if data['DCIS_or_LCIS_여부'].values[0] == 0:
        A = "조직 내에서 DCIS나 LCIS가 관찰되지 않았습니다."
    elif data['DCIS_or_LCIS_여부'].values[0] == 1:
        A = "조직 내에서 DCIS 또는 LCIS가 존재하지만, EIC가 없는 상태입니다."
    elif data['DCIS_or_LCIS_여부'].values[0] == 2:
        A = "조직 내에서 DCIS 또는 LCIS가 존재하며, EIC가 있는 상태이며, 암세포가 유관 내부에서 넓은 영역을 차지하고 있습니다."
    
    Q_list.append(Q)
    A_list.append(A)
    
    if data['DCIS_or_LCIS_여부'].values[0] == 0:
        Q = '조직 내에 DCIS(유관상피내암)나 LCIS(소엽상피내암)가 없어도 전이가 발생할 가능성이 있나요?'
        A = f"""네, DCIS나 LCIS가 조직 내에 관찰되지 않더라도 전이 가능성이 있을 수 있습니다. 전이는 암세포의 침습성과 혈관 또는 림프관 침범 여부에 따라 발생할 수 있기 때문입니다.
        DCIS와 LCIS 부재 시 전이 가능성을 높이는 요인에는 침습성 암(Invasive Carcinoma)의 존재이 존재할 경우, 암세포가 혈관이나 림프관에 침투해서, 원격 전이(간, 폐, 뼈 등)가 발생할 경우,
        높은 증식 속도(KI-67 증가)와 낮은 분화도(HG 3 또는 NG 3)는 전이 가능성을 증가시킬 수 있습니다. 암세포의 공격성과 유전자 돌연변이(HER2 양성 등)가 전이 가능성을 높이는 요인중에 하나입니다. 
        {n_category}"""

        Q_list.append(Q)
        A_list.append(A)
    elif data['DCIS_or_LCIS_여부'].values[0] == 1:
        Q = '조직 내 DCIS 또는 LCIS가 존재하지만 EIC(Extensive Intraductal Component)가 없는 경우 전이 가능성이 있나요 ?'
        A = f"""네, DCIS(유관상피내암) 또는 LCIS(소엽상피내암)가 존재하지만 EIC(Extensive Intraductal Component)가 없는 경우에도 전이 가능성이 있을 수 있습니다. 다만, 그 가능성은 낮은 편이며, 추가적인 병리학적 및 임상적 평가가 필요합니다.
        EIC는 광범위한 상피내 성분(종양의 25% 이상이 상피내 성분으로 이루어진 경우)을 의미하며, 침습성 암과 함께 나타날 경우 예후에 중요한 영향을 미칩니다. 현재 whole slide image에는 EIC가 없는 상태이며, 이는 상피내 병변(DCIS 또는 LCIS)은 존재
        하지만, 침습성 암 내의 상피내 성분이 제한적이라는 것을 의미합니다. 이는 전이 가능성이 비교적 낮음을 시사합니다. 그럼에도 EIC 부재 시에도 전이가 가능한 경우가 존재합니다. DCIS나 LCIS가 혈관이나 림프관에 가까운 위치에 있을 경우, 침습성 암이 없어도, 
        미세 침습(Microinvasion) 상태가 관찰된 경우, DCIS 또는 LCIS가 기저막을 미세하게 침범한 경우, HER2 양성이나 KI-67 증가와 같은 높은 증식 지표를 가진 암세포가 있는 경우, DCIS 또는 LCIS가 조직 경계 근처에 위치하거나 혈관 밀도가 높은 부위에 있을 경우에
        전이 가능성이 있습니다. {n_category}"""
        Q_list.append(Q)
        A_list.append(A)
    elif data['DCIS_or_LCIS_여부'].values[0] == 2:
        Q = '조직 내 DCIS 또는 LCIS가 존재하며, EIC(Extensive Intraductal Component)가 있는 경우 전이 가능성이 있나요?'
        A = f"""네, DCIS(유관상피내암) 또는 LCIS(소엽상피내암)가 존재하며 EIC(Extensive Intraductal Component)가 동반된 경우, 전이 가능성이 현저히 높아질 수 있습니다. EIC는 상피내 병변이 광범위하게 존재하며, 종양의 침습적 특성과 
        암세포의 전이 가능성을 증대시키는 주요 요인으로 간주됩니다. EIC의 임상적으로 침습성 암과 함께 동반되며, 상피내 병변의 광범위한 확산을 암시합니다. 이는 암세포의 침습성, 전이 가능성, 재발률을 증가시킬 수 있습니다. EIC를 동반한 
        DCIS/LCIS는 혈관이나 림프관으로 침범할 가능성이 높아 원격 전이로 이어질 수 있습니다. EIC를 가진 DCIS/LCIS는 높은 KI-67 지표를 보이는 경우가 많아, 세포 분열 활동이 활발하고 전이 가능성을 증가시킵니다. 또한, EIC가 있으면 조직 
        경계가 명확하지 않아, 암세포가 주변 조직으로 확산되기 쉽습니다. {n_category}"""
        Q_list.append(Q)
        A_list.append(A)
    
    return Q_list, A_list

def wsi_diagnosis(data, mode):
    Q_list, A_list = [], []

    nan_check = str(data['NG'].values[0])
    if nan_check != 'nan':
        Q, A = ng_qa(data, mode)
        Q_list += Q
        A_list += A
    
    nan_check = str(data['암의 위치'].values[0])
    if nan_check != 'nan':
        Q, A = tumor_position_qa(data)
        Q_list += Q
        A_list += A
    
    nan_check = str(data['암의 개수'].values[0])
    if nan_check != 'nan':
        Q, A = tumor_counts_qa(data)
        Q_list += Q
        A_list += A
    
    nan_check = str(data['HG'].values[0])
    if nan_check != 'nan':
        Q, A = hg_qa(data, mode)
        Q_list += Q
        A_list += A
    
    nan_check = str(data['DCIS_or_LCIS_여부'].values[0])
    if nan_check != 'nan':
        Q, A= dcis_or_lcis(data)
        Q_list += Q
        A_list += A

    return Q_list, A_list

def text_qa_dataset_gen(data, mode):
    q_list, a_list = [], [] 
    q, a = wsi_diagnosis_all(data, mode)
    q_list.append(q)
    a_list.append(a)

    NG_Q_list, NG_A_list = wsi_diagnosis(data, mode)
    q_list += NG_Q_list
    a_list += NG_A_list
    return q_list, a_list

if __name__ == '__main__':
    #model load
    segmentation_config = '/workspace/whole_slide_image_LLM/wsi_level_vqa-main/utils/wsi_core/segmentation/pidnet_cfg.py'
    segemntation_checkpoint = '/workspace/whole_slide_image_LLM/wsi_level_vqa-main/utils/wsi_core/segmentation/best_mIoU_iter_6400.pth'
    seg_model = init_model(segmentation_config, segemntation_checkpoint)

    pretrained_weights256 = '/workspace/whole_slide_image_LLM/wsi_level_vqa-main/utils/HIPT/checkpoints/vit256_small_dino.pth'
    pretrained_weights4k = '/workspace/whole_slide_image_LLM/wsi_level_vqa-main/utils/HIPT/checkpoints/vit4k_xs_dino.pth'
    device256 = torch.device('cuda')
    device4k = torch.device('cuda')

    model256 = get_vit256(pretrained_weights=pretrained_weights256, device=device256)
    model4k = get_vit4k(pretrained_weights=pretrained_weights4k, device=device4k)
    HIPT_model = HIPT_4K(pretrained_weights256, pretrained_weights4k, device256, device4k)
    HIPT_model.eval()

    img_data_dir = '/workspace/whole_slide_image_LLM/data/train_imgs/'
    meta_data = pd.read_csv("/workspace/whole_slide_image_LLM/data/ori_train.csv")
    wsi_ids = meta_data['ID'].to_list()
    save_dir = "/workspace/whole_slide_image_LLM/data/vqa_dataset/wsi_features/"

    df = pd.DataFrame()
    for i in tqdm(range(len(wsi_ids))):
        wsi_id = wsi_ids[i]
        image_path = os.path.join(img_data_dir, wsi_id + ".png")
        img = cv2.imread(image_path)

        sub_data = meta_data[meta_data['ID'] == wsi_id]
        q_list, a_list = text_qa_dataset_gen(sub_data, 'train')
        features = wsi_representation_features(seg_model, HIPT_model, img)
        features = features.detach().cpu().numpy()
        npy_path = os.path.join(save_dir, wsi_id + ".npy")
        np.save(npy_path, features)
        
        tmp_df = pd.DataFrame({
            "q" : q_list,
            "a" : a_list
        })
        tmp_df['img_feature_path'] = npy_path
        tmp_df = tmp_df[['img_feature_path', 'q', 'a']]
        df = pd.concat([df, tmp_df])
    
    df.to_csv(os.path.join("/workspace/whole_slide_image_LLM/data/vqa_dataset/", 'wsi_vqa_dataset.csv'), index = False)