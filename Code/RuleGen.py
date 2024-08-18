import argparse
import json
import time
from pathlib import Path
import os
import pandas as pd
import random
from AgentSet.UniversalAgent import *
from AgentSet.RoleAgent import *
from AgentSet.RuleStream import *
import PromptSet.Prompts_library as PL
from utils.general_utils import *
from tqdm import tqdm
allcost=0

def clean_json_strings(s):
    str = s.find('[')
    end = s.rfind(']')
    print(f'{s[str:end + 1]}')
    return s[str:end + 1]
time_now = time.strftime("%Y%m%d-%H%M", time.localtime())

def logwrite(GameRecoder,agent_role,response):
    GameRecoder.write("------------------------------\n")
    GameRecoder.write("{} said：{}\n".format(agent_role,response))
    GameRecoder.write("------------------------------\n")

def init_scenarios_v1(args,content,tasks,init_instruction):
    log = open(args.output_folder+ f'{tasks}/init_scenerios.txt', 'a+', encoding='utf-8')  
    interpreter = interpretationer(args.used_model, args.api_path, log)
    #interpreter.user = PL.get_role_prompt+readtxtfile(content)
    interpreter.user = PL.get_role_prompt + content
    interpreter.setMessage()
    while True:
        try:
            roles = interpreter.response_LLMs()
            log.write(roles + '\n')
            roles = clean_json_strings(roles)
            #print(roles)
            roles_txt = json.loads(roles)
            break
        except:
            print('json error,retry！')
    #interpreter.user = PL.get_rule_prompt+readtxtfile(content)
    interpreter.user = PL.get_rule_prompt + content
    interpreter.setMessage()
    while True:
        try:
            rules = interpreter.response_LLMs()
            rules = clean_json_strings(rules)
            #print(rules)
            rules_txt = json.loads(rules)
            break
        except:
            print('json error,retry！')
    roles_list = []
    rules_list = []
    for role in roles_txt:
        temp_log = open(args.output_folder+f'{tasks}/' + role['name'] + '.txt', 'a+', encoding='utf-8')
        temp_role = RoleAgent(role['name'], args.used_model, temp_log, args.api_path, args.WM_num)
        temp_role.set_system("your name is "+role['name']+"\nYour personal information and background are"+role["background"] + "\nYour task is：" + role["task"]+'\nYour decision style is'+PL.decision_style_radical+'\n'+PL.role_prompt)
        #temp_role.receive_message(init_instruction)
        roles_list.append(temp_role)
    # is_init_role=True
    for rule in rules_txt:
        rules_list.append(RuleStream(rule['initiating'],rule['receive'],rule['purpose'],rule['content'],rule['propagation']))
    log.write(rules + '\n')
    return roles_list, rules_list


def init_scenarios_all(args,roles,rules,tasks):
    log = open(args.output_folder + f'log{time_now}.txt', 'a+', encoding='utf-8') 
    interpreter = interpretationer(args.used_model, args.api_path, log)
    interpreter.user =readtxtfile(roles)
    interpreter.setMessage()
    while True:
        try:
            roles = interpreter.response_LLMs()
            print(roles)
            roles = clean_json_string(roles)
            roles=roles.lower()
            roles_txt = json.loads(roles)
            break
        except:
            print('json error,retry！')
    interpreter.user = readtxtfile(rules)
    interpreter.setMessage()
    while True:
        try:
            rules = interpreter.response_LLMs()
            print(rules)
            rules = clean_json_string(rules)
            rules=rules.lower()
            rules_txt = json.loads(rules)
            break
        except:
            print('json error,retry！')
    roles_list = []
    rules_list = []
    for role in roles_txt:
        temp_log = open(args.output_folder + f'{tasks}/' + role['name'] + '.txt', 'a+', encoding='utf-8')
        temp_role = RoleAgent(role['name'], args.used_model, temp_log, args.api_path, args.WM_num)
        temp_role.set_system("your name is " + role['name'] + "\nYour personal information and background are" + role[
            "background"] + "\nYour task is：" + role["task"] + "\nYour decision-making style is:"+PL.decision_style_radical+'\n'+PL.role_prompt)
        # temp_role.receive_message(init_instruction)
        roles_list.append(temp_role)
    for rule in rules_txt:
        rules_list.append(
            RuleStream(rule['initiating'], rule['receive'], rule['purpose'], rule['content'], rule['propagation']))

    return roles_list, rules_list
def init_scenarios(args,roles,rules,tasks):
    roles_list=[]
    rules_list=[]
    with open(roles, 'r', encoding='utf-8') as file:
        roles_txt=file.read()
        roles_txt = clean_json_string(roles_txt)
        roles_txt=json.loads(roles_txt)
    print(roles_txt)
    for role in roles_txt:
        temp_log = open(args.output_folder + f'{tasks}/' + role['name'] + '.txt', 'a+', encoding='utf-8')
        temp_role = RoleAgent(role['name'], args.used_model, temp_log, args.api_path, args.WM_num)
        temp_role.set_system("your name is " + role['name'] + "\nYour personal information and background are" + role[
            "background"] + "\nYour task is：" + role["task"]+"\nYour decision-making style is:"+PL.decision_style_conservative)
        # temp_role.receive_message(init_instruction)
        roles_list.append(temp_role)

    with open(rules,'r',encoding='utf-8') as file:
        rules_txt = file.read()
        rules_txt = clean_json_string(rules_txt)
        rules_txt = json.loads(rules_txt)
    print(rules_txt)
    for rule in rules_txt:
        rules_list.append(
            RuleStream(rule['initiating'], rule['receive'], rule['purpose'], rule['content'], rule['propagation']))
    return roles_list,rules_list
def simulated_scenarios(args,roles,rules,free,log):
    if free==False:
        for rule in rules:
            rule.run(roles,log)
#single scene
def CMGI():
    parser = argparse.ArgumentParser()
    parser.add_argument("-api_path", default='./Key_GPT_0.txt', type=str, help='store api file')
    parser.add_argument("-used_model", default="gpt-3.5-turbo-16k", type=str, help='model type')
    parser.add_argument("-WM_num", default=3, type=int, help='memory space')
    parser.add_argument("-output_folder", default="./outputs/", type=str, help='output path')
    parser.add_argument("-course_topic", default='Dynamic_test', type=str)
    parser.add_argument("-all_messages", default=[], type=str, help='store all message')
    args = parser.parse_args()

 
    args.output_folder = './outputs/{}-{}/'. \
        format(args.course_topic, time_now)
    Path(args.output_folder).mkdir(parents=True, exist_ok=True)  #
    Path(args.output_folder + 'game_theroy').mkdir(parents=True, exist_ok=True)  
    loc_path = args.output_folder+'game_theroy/'
    log=open(loc_path+f'log.txt','a+',encoding='utf-8')
    #roles, rules=init_scenarios_v1(args,scripts[i], tasks=Bias_name[i],init_instruction=init_instructions[i])
    roles, rules = init_scenarios(args, roles='./TxtFile/game_theory/roles.txt',rules='./TxtFile/game_theory/rules.txt', tasks='/game_theroy')
    #roles,rules=init_scenarios(args,roles='./TxtFile/anchor bias/roles.txt',rules='./TxtFile/anchor bias/rules.txt',tasks=None)
    simulated_scenarios(args,roles,rules,False,log)

#multi scene
def CMGI_test():
    parser = argparse.ArgumentParser()
    parser.add_argument("-api_path", default='./Key_GPT_0.txt', type=str, help='store api file')
    parser.add_argument("-used_model", default="gpt-4-turbo", type=str, help='model type')
    parser.add_argument("-WM_num", default=3, type=int, help='memory space')
    parser.add_argument("-output_folder", default="./outputs/", type=str, help='output path')
    parser.add_argument("-course_topic", default='Dynamic test', type=str)
    parser.add_argument("-scripts",default='./TxtFile/dynamic dataset.xlsx')
    parser.add_argument("-all_messages", default=[], type=str, help='store all message')
    args = parser.parse_args()

    #openai.api_base = "https://api.openai.com/v1"
    data=pd.read_excel(args.scripts)
    scripts=data['Script']
    Bias_name=data['Bias Name']
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())

    args.output_folder = './outputs/{}-{}/'. \
        format(args.course_topic, time_now)

    record_format=[]
    record_script=[]


    for i in tqdm(range(0,len(scripts))):
        PL.records_dynatic=[]
        time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
        task=f'{Bias_name[i]} {time_now}'
        Path(args.output_folder).mkdir(parents=True, exist_ok=True) 
        Path(args.output_folder + task).mkdir(parents=True, exist_ok=True) 
        loc_path = args.output_folder + task+'/'
        log=open(loc_path+f'log.txt','a+',encoding='utf-8')
        roles, rules = init_scenarios_v1(args, scripts[i], tasks=task, init_instruction=None)
        simulated_scenarios(args,roles,rules,False,log)
        PL.records_dynatic=json.dumps(PL.records_dynatic)
        record_format.append(PL.records_dynatic)
        record_script.append(scripts[i])
        pd.DataFrame({'record_script':record_script,'record_format':record_format}).to_excel('./res/DB_Note1.xlsx')
    
if __name__ == '__main__':
    CMGI_test()
    #CMGI()