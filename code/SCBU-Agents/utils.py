# encoding: utf-8
"""
@author: Ming Cheng
@contact: ming.cheng@dukekunshan.edu.cn
"""
from config import cfg
import argparse
import io, os, sys, json, logging, cv2
from copy import deepcopy
import soundfile as sf
import numpy as np
import pandas as pd
import h5py
import random
# import torch
import warnings
from pypinyin import pinyin, Style
from config_prompt_pro import cfg_prompt
warnings.filterwarnings('ignore')
from time import sleep
from tqdm import tqdm
from threading import Thread,Lock
import multiprocessing as mp
import requests
import time
import re
formatting_error_query_limit = 3
vpn_timeout_max = 10
set_temperature = 0.7
sleepTime = 15.0
# def set_random_seed(seed, deterministic=True):
#     if seed is not None:
#       random.seed(seed)
#       np.random.seed(seed)
#       torch.manual_seed(seed)
#       torch.cuda.manual_seed_all(seed)
#     if deterministic:
#         torch.backends.cudnn.deterministic = True
#         torch.backends.cudnn.benchmark = False

def get_euclidean_distance(array1, array2):

    array1 = np.array(array1, dtype=np.float)
    array2 = np.array(array2, dtype=np.float)
    
    diff = array1 - array2
    dist = np.sqrt(np.power(diff,2).sum())
    
    return dist


def get_parallel_value(array1, array2):

    array1 = np.array(array1, dtype=np.float)
    array2 = np.array(array2, dtype=np.float)

    # 越接近1说明两向量之间夹角越小越平行，值为0时说明两向量之间为180度相背
    if array1.any() and array2.any():
        norm_a1 = array1 / np.linalg.norm(array1)
        norm_a2 = array2 / np.linalg.norm(array2)
        cos = np.dot(norm_a1, norm_a2)
        val = (np.pi-np.arccos(cos)) / np.pi
    else:
        val = None

    return val

def get_cosine_value(array1, array2):

    array1 = np.array(array1, dtype=np.float)
    array2 = np.array(array2, dtype=np.float)
    
    # 得到的是两个向量的cos(x)的值，越接近1说明越近
    if array1.any() and array2.any():
        norm_a1 = array1 / np.linalg.norm(array1)
        norm_a2 = array2 / np.linalg.norm(array2)
        cos = np.dot(norm_a1, norm_a2)
    else:
        cos = None

    return cos

def save_extracted_imgs(target_dir, img, to_bgr=True):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    name = str(len(os.listdir(target_dir))) + '.png'
    path = os.path.join(target_dir, name)
    if to_bgr:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(path, img)
    

def init_args_parser():
    # 固定随机数种子
    set_random_seed(17)
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default='/data0/yinghuo/邵奕霖')
    parser.add_argument("--config", type=str, default='/home/wxliu/program/codebase3.0/YH_config_h.json')
    args = parser.parse_args()
    update_config(args.config)

    return args
    
        
def update_config(file_path):

    if len(file_path)>0 and os.path.exists(file_path):
        temp_cfg = read_json(file_path)
        for k in temp_cfg.keys():
            if not temp_cfg[k]['valid']:        
                continue
            if temp_cfg[k]['scope']=='all' or temp_cfg[k]['scope']==cfg.name or (cfg.name in temp_cfg[k]['scope']):
                setattr(cfg, k, temp_cfg[k]['param'])
                
    '''
    if hasattr(cfg, 'devices'):
        env = ''
        for d in cfg.devices:
            env += d.split(':')[-1] + ','
        os.environ["CUDA_VISIBLE_DEVICES"] = env

        remapped_devices = []
        for idx in range(len(cfg.devices)):
            remapped_devices.append('cuda:'+str(idx))  
        cfg.devices = remapped_devices 
    '''  
        
    return               
   

def init_output_dir(output_path):
    target_dir = os.path.split(output_path)[0]
    if len(target_dir) > 0 and (not os.path.exists(target_dir)):
        os.makedirs(target_dir, exist_ok=True)
  
    return  

def read_json(file_path):
    """
    从硬盘读取json文件为字典
    Attribtues
        file_path: str, json文件的绝对路径
    Returns
        data: dict, 读取后的字典
    """
    # 读取json文件
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    return data
    
    
def write_json(file_path, data):
    """
    将字典以json文件写入本地
    Attributes
        file_path: str, 将要写入的文件路径
        data: dict, 待写入的数据
    """
    init_output_dir(file_path)
    # 写入json文件
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        
    return None


def init_logger(parad_dir):
    
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    
    # Formatter
    formatter = logging.Formatter('\n\n%(asctime)s - %(filename)s - %(levelname)s - %(message)s \n')

    # File Handler：保存log信息到基础能力的代码库
    file_handler_1 = logging.FileHandler(cfg.local_logger)
    file_handler_1.setFormatter(formatter)
    logger.addHandler(file_handler_1)
    
    # File Handler：保存log信息到分析的范式文件夹
    file_handler_2 = logging.FileHandler(os.path.join(parad_dir, 'logging.log'))
    file_handler_2.setFormatter(formatter)
    logger.addHandler(file_handler_2)
    
    
    if cfg.print_logger:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)  
    
    return logger


def init_video_writer(output_path, w, h, fps, codec, is_color):
    init_output_dir(output_path)
    # init video writer
    fourcc = cv2.VideoWriter_fourcc(*codec)
    VideoWriter = cv2.VideoWriter(output_path, fourcc, fps, (w,h), is_color)  
    
    return VideoWriter


def init_mdata_writer(output_path, default=None):
    
    class MdataWriter():
    
        def __init__(self, output_path, default=None):
            self.output_path = output_path
            if default is None:
                self.results = []
            else:
                self.results = default
                
        def write(self, frame_meta):
            self.results.append(frame_meta)
            
        def insert(self, k, v, frame_idx, instance_idx):
            if frame_idx<len(self.results) and instance_idx<len(self.results[frame_idx]):
                self.results[frame_idx][instance_idx][k] = v
           
        def release(self):
            write_json(self.output_path, self.results)
                 
                 
    init_output_dir(output_path)
    
    return MdataWriter(output_path, default)


def scan_shortest_meta_data(meta_data_list):
    
    lengths = []
    for meta_data in meta_data_list:
        lengths.append(len(meta_data))
        
    if len(lengths) > 0:
        shortest_length = min(lengths)
    else:
        shortest_length = None
        
    return shortest_length
    
def load_intrinsic_data(param_path):
    # read param file
    device_param = read_json(param_path)
    # load color intrin
    if 'color_intrin' in device_param:
        intrinsic = np.identity(3)
        intrinsic[0,0] = device_param['color_intrin']['fx']
        intrinsic[1,1] = device_param['color_intrin']['fy']
        intrinsic[0,2] = device_param['color_intrin']['ppx']
        intrinsic[1,2] = device_param['color_intrin']['ppy']
    else:
        intrinsic = None

    return intrinsic
 
def load_extrinsic_data(param_path):
    # read param file
    device_param = read_json(param_path)
    # load color extrin
    if 'color_extrin' in device_param:
        extrinsic = np.array(device_param['color_extrin'], dtype=np.float64)
    else:
        extrinsic = None
    
    return extrinsic

def load_depth_data(depth_path, frame_idx):
    with h5py.File(depth_path, 'r') as f:
        data = f['depth'][frame_idx]
        
    return data
        
def get_video_length(video_path):
    """
    获取指定视频的总长度
    Attributes
        video_path: 字符串，一个视频文件的绝对路径
    Returns
        length: int类型，指定相机拍摄视频的总长（帧数）
    """
    # 加载视频，获取总帧数后释放资源
    cap = cv2.VideoCapture(video_path)
    length = int(cap.get(7))
    cap.release()

    return length

def get_audio_length(audio_path):
    """
    获取指定音频的总长度
    Attributes
        audio_path: 字符串，一个音频文件的绝对路径
    Returns
        length: int类型，指定音频的总长
    """
    # 读取音频，并获得总长度
    data, rate = sf.read(audio_path)
    length = len(data)

    return length
    
def scan_shortest_video_length(file_manager):
    # 获取所有有效的视频中长度最小的
    lengths = []
    for video_path in file_manager.videos.values():
        if (video_path is not None) and os.path.exists(video_path):
            cur_length = get_video_length(video_path)
            if cur_length > 0:
                lengths.append(cur_length)
                
    if len(lengths) == 0:
        shortest_length = None
    else:
        shortest_length = min(lengths)
    
    return shortest_length

def scan_shortest_audio_length(file_manager):
    lengths = []
    for audio_path in file_manager.audios.values():
        if (audio_path is not None) and os.path.exists(audio_path):
            cur_length = get_audio_length(audio_path)
            if cur_length > 0:
                lengths.append(cur_length)
                
    if len(lengths) == 0:
        shortest_length = None
    else:
        shortest_length = min(lengths)
        
    return shortest_length


def advanced_resize(img, target_size, keep_ratio=False, pad_value=128):

    if keep_ratio: 
        origin_h, origin_w = img.shape[:2]
        target_w, target_h = target_size
        target_w_h_ratio = target_w / target_h
        img = img.copy()
        if (origin_w/origin_h) > target_w_h_ratio:
            # 此时原图过宽，需要pad上下两侧
            pad_size = (origin_w - target_w_h_ratio * origin_h) / target_w_h_ratio
            pad_half = int(pad_size/2)
            pad_data = np.ones((pad_half,origin_w,3), dtype=np.uint8) * int(pad_value)
            img = np.concatenate([pad_data,img,pad_data], axis=0)
            #reshaped_img = cv2.resize(reshaped_img, (target_w,target_h)) 
        else:
            # 此时原图过窄，需要pad左右两侧
            pad_size = target_w_h_ratio * origin_h - origin_w 
            pad_half = int(pad_size/2)
            pad_data = np.ones((origin_h,pad_half,3), dtype=np.uint8) * int(pad_value)
            img = np.concatenate([pad_data, img,pad_data], axis=1)
            #reshaped_img = cv2.resize(reshaped_img, (target_w,target_h)) 

    reshaped_img = cv2.resize(img, target_size)

    return reshaped_img

def get_squared_bbox(bbox):

    x1,y1,x2,y2 = bbox[0],bbox[1],bbox[2],bbox[3]
    w = x2 - x1
    h = y2 - y1
    if w < h:
        gap = round((h-w) / 2)
        x1 -= gap
        x2 += gap
    else:
        gap = round((w-h) / 2)
        y1 -= gap
        y2 += gap


    return [x1,y1,x2,y2]

def correct_coord_outliers(coords, origin_res):
    new_coords = deepcopy(coords)
    for idx, coord in enumerate(coords):
        if idx%2 == 0:    
            new_x = min(max(0, coords[idx]), origin_res[0]-1)
            new_coords[idx] = new_x
        if idx%2 == 1:
            new_y = min(max(0, coords[idx]), origin_res[1]-1)
            new_coords[idx] = new_y

    return new_coords


def get_squared_image(img, bbox, target_size=None):

    h,w = img.shape[:2]

    squared_bbox = get_squared_bbox(bbox)
    squared_bbox = correct_coord_outliers(squared_bbox, (w,h))
    x1,y1,x2,y2  = squared_bbox

    cropped_img = img[y1:y2,x1:x2]
    if len(cropped_img.shape)!=3 or (0 in cropped_img.shape):
        cropped_img = np.zeros((224,224,3), dtype=np.uint8) 

    if target_size is None:
        max_size = max(cropped_img.shape[:2])
        target_size = (max_size, max_size)

    squared_img = advanced_resize(img=cropped_img, target_size=target_size, keep_ratio=True, pad_value=128) 

    return squared_img
    
def get_img_from_fig(fig, dpi=180, img_size=None):
    fig.canvas.draw()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches='tight', dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    if img_size is not None:
        img = cv2.resize(img, img_size)
    
    return img


###########################################################################
#prompt 构建需要函数
def frameIndexToStr(time,fps=8):
    second = int(time) / fps
    minute = int(second) // 60
    secondInt = int(second) % 60
    secondFloat = second - minute*60 - secondInt
    secondStr = str(minute).zfill(2) +':'+ str(secondInt).zfill(2) + '.' + str(int(secondFloat*1000)).zfill(3)

    return secondStr

#time1,time2 : int or str
def FrameRangeToStr(time1,time2):
    out = frameIndexToStr(time1)+'-'+ frameIndexToStr(time2)+': '
    return out 

def save_to_txt(data, file_path):
    # 打开文件，如果文件不存在则会创建
    init_output_dir(file_path)
    with open(file_path, 'w') as f:
        # 将数据写入文件
        f.write(data)


def read_from_txt(file_path):
    # 打开文件，如果文件不存在则会报错  
    with open(file_path, 'r') as f: 
        # 读取文件内容并返回    
        data = f.read()
    return data

#计算历史时间
def cul_history_time(name,paradigm_name, save_root='./results'):
    parad_list = [
                'a1', 'a2', 
                'b1', 'b2', 'b3', 'b4', 
                'c1', 'c2', 'c3', 'c4', 
                'd1', 
                'e1', 'e2', 'e3', 
                'g1', 'g2'] 
    
    total_time = 0
    for item in parad_list:
        if item == paradigm_name:
            break
        cur_path= os.path.join(save_root,name,item+'.json')
        if not os.path.exists(cur_path):
            continue
        data = read_json(cur_path)
        all_time =  int(data['total_frame_nums'])
        total_time +=  all_time
    return total_time


def cul_history_time2(name,paradigm_name, save_root='./results'):
    parad_list = [
                'a1', 'a2', 
                'b1', 'b2', 'b3', 'b4', 
                'c1', 'c2', 'c3', 'c4', 
                'd1', 
                'e1', 'e2', 'e3', 
                'g1', 'g2'] 
    
    total_time = 0
    for item in parad_list:
        if item == paradigm_name:
            break
        cur_path= os.path.join(save_root,name,item+'.json')
        if not os.path.exists(cur_path):
            continue
        data = read_json(cur_path)
        all_time =  int(data['total_frame_nums'])
        total_time +=  all_time + cfg_prompt.preparatory_time
    return total_time

def read_medical_record(path):
    record = pd.read_excel(path, index_col='name', parse_dates=['birthday', 'test-date'],
             usecols=['name', 'gender', 'birthday', 'test-date', 'label'])
    for i in record.index:
        raw_data = record.loc[i].tolist()
        non_data = [i for i in raw_data if pd.isnull(i)] 
        if len(non_data)>3:
            record = record.drop(index=i)
    return record


def read_protocol_scores(path, usecols=None):
    scores = pd.read_excel(path, index_col='name', usecols=usecols)
    for i in scores.index:
        raw_data = scores.loc[i].tolist()
        non_data = [i for i in raw_data[:16] if pd.isnull(i)] # 只看前16个内容（范式得分项目）
        if len(non_data)>14:
            scores = scores.drop(index=i)
    return scores

def clean_database(database, predicted_label='label', dropped_labels=[]):
    
    database = database.copy()
    for i in database.index:
        sample = database.loc[i]
        if pd.isnull(sample[predicted_label]):
            database = database.drop(index=i)

    # 出去一些不要的类别   
    for i in database.index:
        sample = database.loc[i]
        if sample[predicted_label] in dropped_labels:
            database = database.drop(index=i)

    return database

def chinese_to_pinyin(text, style=Style.NORMAL):
    return ''.join([y[0] for y in pinyin(text, style=style)])
 
#根据字典中的vaule值排序
def sortedDictFromVaule(mydict,flag=False):
    sorted_items = sorted(mydict.items(), key=lambda item: item[1],reverse =flag)
    sorted_dict = {k: v for k, v in sorted_items}
    return sorted_dict

def isPath(path):
    if not os.path.exists(path):
        os.makedirs(path)

class MPCount():
    
    def __init__(self, total):
        self.cur_steps = 0
        self.all_steps = total
        self.lock = Lock()
        self.stop = False
        self.still_listen = True
        self.still_update = True

        # queues that can be shared between async processes
        self.m = mp.Manager()
        self.queue = self.m.Queue()
        self.error = self.m.Queue()
        # 开始计数
        self.begin()

        return

    def begin(self):
        t1 = Thread(target=self.listen, daemon=True)
        t1.start()
        t2 = Thread(target=self.update, daemon=True)
        t2.start()
        return 
        
    def close(self):
        self.lock.acquire()
        self.stop = True
        self.lock.release()
        while self.still_listen or self.still_update:
            sleep(1)

        # self.successChilds.close()
        # self.failChilds.close()

        return
        
    def listen(self):
        while True:
            sleep(1)
            while not self.queue.empty():
                data = self.queue.get()
                if data is not None:
                    self.lock.acquire()
                    self.cur_steps += int(data)
                    self.lock.release()  
            if self.stop and self.queue.empty():
                break
        self.lock.acquire()
        self.still_listen = False
        self.lock.release()
        return 

    # def saveinfo(self,child_name,flag=True):
    #     if flag:
    #         self.successChilds.write(child_name)
    #         self.successChilds.write("\n")   
    #     else:
    #         self.failChilds.write(child_name)
    #         self.failChilds.write("\n")
    #     return

    def update(self):
        show_steps = 0
        with tqdm(total=self.all_steps, ncols=70) as pbar:
            while True:  
                sleep(1)
                # 更新进度条并休眠等待
                self.lock.acquire()
                pbar.update(self.cur_steps - show_steps)
                show_steps = self.cur_steps
                self.lock.release()   
                # 最后一次更新进度条后退出
                if self.stop and (not self.still_listen):
                    self.lock.acquire()
                    pbar.update(self.cur_steps - show_steps)
                    show_steps = self.cur_steps
                    self.lock.release() 
                    break                  
                    
        self.lock.acquire()
        self.still_update = False
        self.lock.release()
        return
    
def find_structured_data(data):
    reasons = re.search(r'1\. Reasons for autism spectrum disorder judgment:\s*\[(.*?)\]', data, re.DOTALL)
    results = re.search(r'2\. Autism spectrum disorder Judgment results:\s*\[(.*?)\]', data, re.DOTALL)
    persuasion = re.search(r'3\. Persuasion:\s*(.*)', data, re.DOTALL)
    flag = True
    if reasons:
        reasons = reasons.group(1).strip()
    else:
        reasons = '' 
        flag = False

    if results:
        results = results.group(1).strip()
    else:
        results = ''
        flag = False

    if persuasion:
        persuasion = persuasion.group(1)
        persuasion = persuasion.strip()
    else:
        persuasion = ''
        flag = False
    return flag,reasons,results,persuasion

#调用openai kpi
def sendOneMessageToOpenAI(text_prompt,synthesizer, save_path,token_number = 1000,check_flag = True):
    send_prompt = {
        "model" : 'gpt-4o',
        "messages": [
            {"role": "system", "content":synthesizer},
            {"role": "user" , "content": text_prompt}
        ],
        "max_tokens" : token_number,
        "temperature" : 0,
    }

    data = json.dumps(send_prompt)

    base_url = 'https://api.openai.com'
    url = base_url + "/v1/chat/completions"
    API_KEY = 'your_api_key'

    headers = {"Authorization": f"Bearer {API_KEY}",\
                "Content-Type" : "application/json",\
               }
    # respone = requests.post(url,headers=headers,data=data)
    # response = respone.json()
    # output_content = response["choices"][0]["message"]["content"]
    # time.sleep(sleepTime)
    # #print(output_content)
    # 
    # struction_flag,reasons,results,persuasion = find_structured_data(data)

    #检查输出格式
    #如果不符合进行重复询问
   
    struction_flag = False
    query_count =0
    while struction_flag == False and query_count < formatting_error_query_limit:
        print('gpt4o is running')
        for i in range(vpn_timeout_max):
            try:
                respone = requests.post(url,headers=headers,data=data)
                break
            except:
                print(f"openai 第{i+1}次请求失败：")
                time.sleep(sleepTime)
        response = respone.json()
        output_content = response["choices"][0]["message"]["content"]
        write_json(save_path,response)
        time.sleep(sleepTime)
        if check_flag:
            struction_flag,reasons,results,persuasion = find_structured_data(output_content)
        else:
            struction_flag = True
        query_count+=1

def sendOneMessageTollama3(llama3_api,text_prompt,synthesizer, save_path,token_number = 1000):

    data = [
          {"role": "system", "content":synthesizer},
          {"role": "user" , "content": text_prompt}
    ]
    chat_completion = llama3_api.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-70B-Instruct",
        messages= data,
        stream=False,
    )
    #print(chat_completion.choices[0].message.content)

    response = dict(chat_completion.choices[0].message)
    write_json(save_path,response)
    time.sleep(sleepTime)

def sendOneMessageTodeepseek(deepseek_api,text_prompt,synthesizer, save_path,token_number = 1000):
    #检查输出格式
    #如果不符合进行重复询问
    struction_flag = False
    query_count =0
    while struction_flag == False and query_count < formatting_error_query_limit:
        print('deepseekr1 is running')
        chat_completion = deepseek_api.chat.completions.create(
        model="deepseek-chat",
        messages=[
            {"role": "system", "content": synthesizer},
            {"role": "user", "content": text_prompt},
        ],
        stream=False,
        temperature = 0,
        max_tokens= token_number
        )
        #print(chat_completion.choices[0].message.content)
        response = dict(chat_completion.choices[0].message)
        write_json(save_path,response)
        time.sleep(sleepTime)
        text = response["content"]
        output_content = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
        struction_flag,reasons,results,persuasion = find_structured_data(output_content)
        query_count+=1
       

def sendOneMessageToqwen(qwen_api,text_prompt,synthesizer, save_path,token_number = 1000):
    data = [
          {"role": "system", "content":synthesizer},
          {"role": "user" , "content": text_prompt}
    ]

    #检查输出格式
    #如果不符合进行重复询问
    struction_flag = False
    query_count =0
    while struction_flag == False and query_count < formatting_error_query_limit:
        print('qwen2 is running')
        for i in range(vpn_timeout_max):
            try:
                chat_completion = qwen_api.chat.completions.create(
                    model="Qwen/Qwen2.5-72B-Instruct",
                    messages= data,
                    max_tokens = token_number,
                    temperature = 0
                )
            except :
                print(f"openai 第{i+1}次请求失败：")
                time.sleep(sleepTime)
        #print(chat_completion.choices[0].message.content)
        response = dict(chat_completion.choices[0].message)
        write_json(save_path, response)
        time.sleep(sleepTime)
        output_content = response["content"]
        struction_flag,reasons,results,persuasion = find_structured_data(output_content)
        query_count+=1

def sendOneMessageTocluade(text_prompt,synthesizer, save_path,token_number = 1000):
    send_prompt = {
        "model": "claude-3-5-sonnet-20240620",
        "system" :synthesizer,
        "messages": [
            {"role": "user", "content": text_prompt}
        ],
        "max_tokens": token_number,
        "temperature": 0
    }
    base_url = "https://api.anthropic.com"
    data = json.dumps(send_prompt)
    url = base_url + "/v1/messages"
    headers = {"x-api-key": "your_api_key", \
               "content-type": "application/json", \
               "anthropic-version":'2023-06-01'
               }
    
    #检查输出格式
    #如果不符合进行重复询问
    struction_flag = False
    query_count =0
    while struction_flag == False and query_count < formatting_error_query_limit:
        print('claude3_5 is running')
        for i in range(vpn_timeout_max):
            try:
                response = requests.post(url, headers=headers, data=data)
                break
            except:
                print(f"cluade 第{i+1}次请求失败：")
                time.sleep(sleepTime)

        response = response.json()
        #print(response)
        output_content = response["content"][0]["text"]
        #print(output_content)
        write_json(save_path, response)
        time.sleep(sleepTime)
        struction_flag,reasons,results,persuasion = find_structured_data(output_content)
        query_count+=1

#kimi
def sendOneMessageTokimi(text_prompt,synthesizer, save_path,token_number = 1000):
    send_prompt = {
        "model": 'moonshot-v1-32k',
        "messages": [
            {"role": "system", "content": synthesizer},
            {"role": "user", "content": text_prompt}
        ],
        "max_tokens": token_number,
        "temperature": 0
    }
    base_url = 'https://api.moonshot.cn'
    api_key ='your_api_key'

    data = json.dumps(send_prompt)
    url = base_url + "/v1/chat/completions"

    headers = {"Authorization": f"Bearer {api_key}",\
                "Content-Type" : "application/json",\
               }
    
    #检查输出格式
    #如果不符合进行重复询问
    struction_flag = False
    query_count =0
    while struction_flag == False and query_count < formatting_error_query_limit:
        print('kimi is running')
        response = requests.post(url,headers=headers,data=data)
        response = response.json()
        output_content = response["choices"][0]["message"]["content"]
        #print(output_content)
        write_json(save_path, response)
        time.sleep(sleepTime)
        struction_flag,reasons,results,persuasion = find_structured_data(output_content)
        query_count+=1

#国内代理商
def sendOneMessageTocluade2(cluade_api,text_prompt,synthesizer, save_path,token_number = 1000):
    chat_completion = cluade_api.chat.completions.create(
    model="claude-3-5-sonnet-20241022", #填写claude模型名称即可
    messages=[
        {
            "role": "user",
            "content": "Hello!"
        }
    ]
    )

    # 打印返回结果
    print(chat_completion.choices[0].message.content)
    response = dict(chat_completion.choices[0].message)
    write_json(save_path,response)
    time.sleep(sleepTime)