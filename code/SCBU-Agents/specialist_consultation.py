# -*- coding: utf-8 -*-
# @Author: Your name
# @Date:   2025-03-27 12:20:12
# @Last Modified by:   Your name
# @Last Modified time: 2025-04-19 21:30:03
#专家会诊
import re
from utils import read_json,write_json,read_from_txt,sendOneMessageToOpenAI,sendOneMessageTollama3,sendOneMessageTodeepseek,sendOneMessageTocluade,sendOneMessageToqwen,sendOneMessageTokimi
from generateActionJson import getDatabasePaths
from collections import Counter
from openai import OpenAI
import os
import sys
from metrics import *
from config import cfg
#读取映射表
IndexToName = read_json('IndexToName.json')
NameToIndex = read_json('NameToIndex.json')


class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

sys.stdout = Logger("./output_emotion.log")

def parse_Judgment(content):
    search = re.search('\[(\d)\]',content)
    score = 1
    if search:
        score = int(search.group(1))
    else:
        search = re.search('Autism spectrum disorder Judgment results:[\\w\\s\\n]*(\d)',content)
        if search:
            score = int(search.group(1))

    if score not in [0,1]:
        score = 1
    return score
    
def read_model_json(json_path, model_name):
    response = read_json(json_path)
    if model_name == 'gpt4o':
        output_content = response["choices"][0]["message"]["content"]
    elif model_name == 'claude3_5':
        output_content = response["content"][0]["text"]
    elif model_name == 'deepseekr1_671B':
        text = response["content"]
        output_content = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    elif model_name == 'llama3':
        output_content = response["content"]
    elif model_name == 'qwen2':
        output_content = response["choices"][0]["message"]["content"]
    elif model_name == 'kimi':
        output_content = response["choices"][0]["message"]["content"]
    score = parse_Judgment(output_content)
    return output_content,score

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
    
def read_model_json_debate(json_path, model_name):
    response = read_json(json_path)
    if model_name == 'gpt4o':
        output_content = response["choices"][0]["message"]["content"]
    elif model_name == 'claude3_5':
        output_content = response["content"][0]["text"]
    elif model_name == 'deepseekr1_671B':
        text = response["content"]
        output_content = re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()
    elif model_name == 'llama3':
        output_content = response["content"]
    elif model_name == 'qwen2':
        output_content = response["content"]
    elif model_name == 'kimi':
        output_content = response["choices"][0]["message"]["content"]

    struction_flag,reasons,results,persuasion = find_structured_data(output_content)

    output_content = f'1. Reasons for autism spectrum disorder judgment: [{reasons}]\n 2. Autism spectrum disorder Judgment results: [{results}]\n'

    #persuasion = f'3. Persuasion: [{persuasion}]\n'

    score = parse_Judgment(output_content)
    return struction_flag,output_content,persuasion,score

def summarization_report(name,scripts_path,init_specialist_diagnostics_root):
    
    scripts = read_from_txt(scripts_path.format(name))
    summarized_report ='<Script> \n' + scripts + '\n </Script>\n'
        
    specialist_names =['gpt4o','claude3_5','kimi','qwen2','deepseekr1_671B']
    summarized_report +=  '<Summary of diagnostic results>\n'

    for count in range(len(specialist_names)):
        model_name = specialist_names[count]
        init_specialist_diagnostics_path = init_specialist_diagnostics_root.format(name,model_name)
        output_content,score = read_model_json(init_specialist_diagnostics_path,model_name)
        summarized_report += f'<doctor {str(count+1)}>\n' + output_content + f'\n</doctor {str(count+1)}>\n'

    summarized_report +=  '<Summary of diagnostic results>'

    return summarized_report


def summarization_report_debate(name,scripts_path,round_index,debate_root_format):
    debate_root = debate_root_format.format(name, str(round_index))
    scripts = read_from_txt(scripts_path.format(name))
    summarized_report ='<Script> \n' + scripts + '\n </Script>\n'

    specialist_names =['gpt4o','claude3_5','kimi','qwen2','deepseekr1_671B']
    summarized_report +=  '<Summary of diagnostic results>\n'

    for count in range(len(specialist_names)):
        model_name = specialist_names[count]
        debate_path = os.path.join(debate_root,model_name+'.json')
        struction_flag,output_content,persuasion,score = read_model_json_debate(debate_path,model_name)
        summarized_report += f'<doctor {str(count+1)}>\n' + output_content + f'\n</doctor {str(count+1)}>\n '

    summarized_report +=  '</Summary of diagnostic results>\n'

    return summarized_report


def summarization_report_add_debate(name,model_name_number,scripts_path,last_debate_round_index,debate_root_format):
    debate_root = debate_root_format.format(name, str(last_debate_round_index))
    scripts = read_from_txt(scripts_path.format(name))
    summarized_report ='<Script> \n' + scripts + '\n </Script>\n'

    specialist_names =['gpt4o','claude3_5','kimi','qwen2','deepseekr1_671B']
    summarized_report +=  '<Summary of diagnostic results>\n'

    for count in range(model_name_number):
        model_name = specialist_names[count]
        debate_path = os.path.join(debate_root,model_name+'.json')
        struction_flag,output_content,persuasion,score = read_model_json_debate(debate_path,model_name)
        #print(model_name, 'struction_flag:',struction_flag)
        summarized_report += f'<doctor {str(count+1)}>\n' + output_content + f'\n</doctor {str(count+1)}>\n '

    summarized_report +=  '</Summary of diagnostic results>\n'

    summarized_report_adddebate = summarized_report +  '<debate>\n'
    for count in range(model_name_number):
        model_name = specialist_names[count]
        debate_path = os.path.join(debate_root,model_name+'.json')
        struction_flag,output_content,persuasion,score = read_model_json_debate(debate_path,model_name)
        summarized_report_adddebate = summarized_report_adddebate + persuasion + '\n'
    summarized_report_adddebate +=  '<\debate>\n'

    return summarized_report,summarized_report_adddebate
    
    
def get_maker_prompt(summarized_report):
    synthesizer = 'You are a medical decision maker in the field of autism spectrum disorder and are skilled at summarizing and generalizing opinions based on input from multiple autism experts.'
    
    text_prompt = summarized_report + "\n" + "The <Script> is a chronological log of the behavior of a child with suspected autism spectrum disorder." \
                 "The <Summary of diagnostic results> contains the diagnostic results and diagnostic rationale for the <Script> from multiple autism experts. "\
                 "The judgment result in <doctor> is a number, 0 means Typical Development (TD), and 1 means autism spectrum disorder (ASD). \n"\
                 "You need to complete the following steps: " \
                 "1. Carefully and comprehensively consider <script> and <Summary of diagnostic results>."\
                 "2. Count the opinions of the doctors and determine whether all of them are in agreement." \
                 "3. Summarize the reasons for all the doctors\' judgments, with the ultimate goal of forming a concise diagnostic rationale."\
                 "4. You should output in exactly the same format as '''1. Consensus: [ ] (ASD or TD). 2. Reasons: [ ]''' "
    
    return synthesizer, text_prompt

def get_specialist_prompt(index,summarized_report):
    synthesizer = f'You are an autism spectrum disorder diagnostic specialist, You know all the symptoms and criteria for judging Autism Spectrum Disorder. Your specialist number is <doctor {index}>. '

    text_prompt = summarized_report + "\n" + f" The <Script> is a chronological log of the behavior of a child with suspected autism spectrum disorder. " \
                f"The <Summary of diagnostic results> contains the diagnostic results and diagnostic rationale for the <Script> from multiple autism experts. "\
                f"The judgment result in <doctor> is a number, 0 means Typical Development (TD), and 1 means autism spectrum disorder (ASD). \n"\
                f"You need to complete the following steps: " \
                f"1. Carefully and comprehensively consider <script>. " \
                f"2. Consider revising your own diagnosis in the <doctor {index}> by taking into account the <Summary of diagnostic results>. Based on this analysis, re-generate the diagnosis and provide a rationale." \
                f"3. Attempt to convince other experts who hold a different view, arguing with specific examples from the <Script>. " \
                f"4. You should output in exactly the same format as '''1. Reasons for autism spectrum disorder judgment: [ ]\n 2. Autism spectrum disorder Judgment results: [ ] (0 or 1)\n 3. Persuasion: [<doctor {index}> to <doctor index>: ]  (You can convince one expert or more than one expert, all following this format)  ''' " 

    return synthesizer, text_prompt


def get_specialist_prompt_debate(index,summarized_report):
    synthesizer = f'You are an autism spectrum disorder diagnostic specialist, You know all the symptoms and criteria for judging Autism Spectrum Disorder. Your specialist number is <doctor {index}>. '

    text_prompt = summarized_report + "\n" + f" The <Script> is a chronological log of the behavior of a child with suspected autism spectrum disorder. " \
                f"The <Summary of diagnostic results> contains the diagnostic results and diagnostic rationale for the <Script> from multiple autism experts. "\
                f"The <debate> records the process of debate between multiple experts. Format as [<doctor {index}> to <doctor index>:]. "\
                f"The judgment result in <doctor> is a number, 0 means Typical Development (TD), and 1 means autism spectrum disorder (ASD). \n"\
                f"You need to complete the following steps: " \
                f"1. Carefully and comprehensively consider <script>. " \
                f"2. Think carefully about the process of debate between multiple experts in <debate>, especially the words of other experts to yourself. Please note that the words of other experts may not be entirely trustworthy."\
                f"2. Consider revising your own diagnosis in the <doctor {index}> by taking into account the <Summary of diagnostic results> and the <debate>. Based on this analysis, re-generate the diagnosis and provide a rationale." \
                f"3. Attempt to convince other experts who hold a different view, arguing with specific examples from the <Script> and the <debate>. " \
                f"4. You should output in exactly the same format as '''1. Reasons for autism spectrum disorder judgment: [ ]\n 2. Autism spectrum disorder Judgment results: [ ] (0 or 1)\n 3. Persuasion: [<doctor {index}> to <doctor index>: ] (You can convince one expert or more than one expert, all following this format) ''' " 

    return synthesizer, text_prompt



def first_round(child_name_index,scripts_path,init_specialist_diagnostics_root,save_root_old):
    child_name = IndexToName[str(child_name_index)]
    model_name_number = 5
    summarized_report = summarization_report(child_name, scripts_path, init_specialist_diagnostics_root)
    
    save_root =  os.path.join(save_root_old,child_name,'round1')    #f'./result_agent/{child_name}/round1'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    print(f'Round1: subject {child_name_index} is processing:')
    ## openai
    synthesizer, text_prompt = get_specialist_prompt(1,summarized_report)
    save_path = os.path.join(save_root,'gpt4o.json')
    sendOneMessageToOpenAI(text_prompt,synthesizer,save_path,1500)

    #国外cluade3.5
    synthesizer, text_prompt = get_specialist_prompt(2,summarized_report)
    save_path = os.path.join(save_root,'claude3_5.json')
    sendOneMessageTocluade(text_prompt,synthesizer,save_path,1500)

    #kimi
    synthesizer, text_prompt = get_specialist_prompt(3,summarized_report)
    save_path = os.path.join(save_root,'kimi.json')
    sendOneMessageTokimi(text_prompt,synthesizer,save_path,1500)

    #qwen2 api
    infra_api = OpenAI(
    api_key="your_api_key",
    base_url="https://api.deepinfra.com/v1/openai",)
    synthesizer, text_prompt = get_specialist_prompt(4,summarized_report)
    save_path = os.path.join(save_root,'qwen2.json')
    sendOneMessageToqwen(infra_api,text_prompt,synthesizer,save_path,1500)

    #deepseekr1
    synthesizer, text_prompt = get_specialist_prompt(5,summarized_report)
    deepseekr1 = OpenAI(api_key="your_api_key", base_url="https://api.deepseek.com")
    save_path = os.path.join(save_root,'deepseekr1_671B.json')
    sendOneMessageTodeepseek(deepseekr1,text_prompt,synthesizer,save_path,1500)

def debate_round(round_index,child_name_index,scripts_path,debate_root_format,old_save_path):
    child_name = IndexToName[str(child_name_index)] 
    model_name_number = 5 
    last_debate_round_index = round_index-1 
    #debate_root_format = './result_agent/{}/round{}'
    summarized_report,summarized_report_adddebate = summarization_report_add_debate(child_name,model_name_number,scripts_path,last_debate_round_index,debate_root_format)
    #print(summarized_report_adddebate)
    
    save_root = os.path.join(old_save_path,child_name,f'round{round_index}')#f'./result_agent/{child_name}/round{round_index}'
    if not os.path.exists(save_root):
        os.makedirs(save_root)
    #print(f'Round{round_index}: subject {child_name_index} is processing:')

    ## openai
    synthesizer, text_prompt = get_specialist_prompt_debate(1,summarized_report_adddebate)
    save_path = os.path.join(save_root,'gpt4o.json')
    sendOneMessageToOpenAI(text_prompt,synthesizer,save_path,1500)

    #国外cluade3.5
    synthesizer, text_prompt = get_specialist_prompt_debate(2,summarized_report)
    save_path = os.path.join(save_root,'claude3_5.json')
    sendOneMessageTocluade(text_prompt,synthesizer,save_path,1500)

    #kimi
    synthesizer, text_prompt = get_specialist_prompt_debate(3,summarized_report)
    save_path = os.path.join(save_root,'kimi.json')
    sendOneMessageTokimi(text_prompt,synthesizer,save_path,1500)

    #qwen2 api
    infra_api = OpenAI(
    api_key="your_api_key",
    base_url="https://api.deepinfra.com/v1/openai",)
    synthesizer, text_prompt = get_specialist_prompt_debate(4,summarized_report)
    save_path = os.path.join(save_root,'qwen2.json')
    sendOneMessageToqwen(infra_api,text_prompt,synthesizer,save_path,1500)

    #deepseekr1
    synthesizer, text_prompt = get_specialist_prompt_debate(5,summarized_report)
    deepseekr1 = OpenAI(api_key="your_api_key", base_url="https://api.deepseek.com")
    save_path = os.path.join(save_root,'deepseekr1_671B.json')
    sendOneMessageTodeepseek(deepseekr1,text_prompt,synthesizer,save_path,1500)

#检查共识
def checkFinalResults(round_index,name,init_specialist_diagnostics_root,debate_root):
    specialist_names =['gpt4o','claude3_5','kimi','qwen2','deepseekr1_671B']
    struction_flag = True
    all_score = []
    struction_flags = []
    if round_index == 0:
        for count in range(len(specialist_names)):
            model_name = specialist_names[count]
            init_specialist_diagnostics_path = init_specialist_diagnostics_root.format(name,model_name)
            output_content,score = read_model_json(init_specialist_diagnostics_path,model_name)
            all_score.append(score)
    else:
        for count in range(len(specialist_names)):
            model_name = specialist_names[count]
            debate_root = debate_root.format(name,round_index)#f'./result_agent/{name}/round{round_index}'
            debate_path = os.path.join(debate_root,model_name+'.json')
            struction_flag,output_content,persuasion,score = read_model_json_debate(debate_path,model_name)
            all_score.append(score)
            struction_flags.append(struction_flag)
    
    flag = False
    if len(set(all_score)) == 1 :
        flag = True
    else:
        flag = False

    if len(set(struction_flags)) == 1:
        struction_flag = True
    else:
        struction_flag = False 

    return flag, all_score, struction_flag,struction_flags

#得到最终诊断共识
def getConsensus(round_index,child_name,scripts_path,init_specialist_diagnostics_root,debate_root_format,save_root,check_flag=True):
    
    if round_index == 0:
        summarized_report = summarization_report(child_name,scripts_path,init_specialist_diagnostics_root)
    else:
        summarized_report = summarization_report_debate(child_name,scripts_path,round_index,debate_root_format)
    synthesizer, text_prompt = get_maker_prompt(summarized_report)
    save_path = os.path.join(save_root,child_name,'consensus.json')
    sendOneMessageToOpenAI(text_prompt,synthesizer,save_path,1500,check_flag)

def agentDebate(child_name_index):
    #w/o emotion
    # scripts_path = './results_prompt_pro/{}/context.txt'
    # init_specialist_diagnostics_root  = './results_finnal/{}/best_{}.json'
    # debate_root_format = './result_agent/{}/round{}'
    #  save_root= './result_agent/'

    # w emotion
    scripts_path = './results_addEmotion2s_filter_0.175/{}/context.txt'
    init_specialist_diagnostics_root  = './results_finnal_withemotion/{}/best_{}.json'
    debate_root_format = './result_agent_emotion/{}/round{}'
    save_root= './result_agent_emotion/'
    max_round = 5
    child_name = IndexToName[str(child_name_index)]
    max_flag = True
    print(f'********************************** subject {child_name_index}: *****************************************')
    for _round_index in range(max_round):
        flag, all_score, struction_flag,struction_flags= checkFinalResults(_round_index,child_name,init_specialist_diagnostics_root,debate_root_format)
        if _round_index !=0:
            print(f'check round{str(_round_index)}: output format is {struction_flag}, detail is {struction_flags}')
        #达成一致，做出最后决定
        if flag:
            print(f'********** round{_round_index} reach consensus:{all_score}! ************')
            getConsensus(_round_index,child_name,scripts_path,init_specialist_diagnostics_root,debate_root_format,save_root,False)
            max_flag =False
            break
        else:        
            print(f'############ round {_round_index+1} processing....###################')
            if _round_index == 0:
                first_round(child_name_index,scripts_path,init_specialist_diagnostics_root,save_root)
            else:
                debate_round(_round_index+1,child_name_index,scripts_path,debate_root_format,save_root)

    if max_flag:
        print(f'********** round{_round_index} reach consensus! ************')
        getConsensus(_round_index,child_name,scripts_path,init_specialist_diagnostics_root,debate_root_format,save_root,False)


#计算准确率
def calculateAccuracy(result_root='./result_agent/'):
    paths = getDatabasePaths(cfg.root)
    count = 0
    right = 0
    failNames =[]
    failNames_pinyin =[]
    pred = []
    ture = []
    for k,v in paths.items():
        json_path = os.path.join(result_root,k,'consensus.json')
        gt = v['label']
        response = read_json(json_path)
        output_content = response["choices"][0]["message"]["content"]
        search = re.search('Consensus: \s*\[(.*?)\]',output_content)
        score = 1
        if search:
            score = search.group(1)
            if score =='ASD' or score =='1':
                score = 1
            else:
                score = 0
        else:
            print(f'{k} not find result')   

        if score not in [0,1]:
            score = 1
        
        ture.append(gt)
        pred.append(score)
        if score == gt :
            right +=1
        else:
            failNames.append(k)
            failNames_pinyin.append(v['name_pinyin'])
        count +=1
    #print('acc :', round(right/count,4))
    #print('failNames:',failNames_pinyin)
    print_classification_report(ture,pred,[0,1],'binary',True)
    print('failNames:',failNames)
    plot_confusion_matrix(ture, pred, 
                      labels=[0,1], normalize=None)
    return failNames  

if __name__ == '__main__':

    agentDebate(0)
  