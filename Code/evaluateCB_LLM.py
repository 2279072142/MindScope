import json
import re
import pandas as pd
import os
import argparse
import time
import openai
from tqdm import tqdm
from PromptSet.prompt import *
from LLM.base_openai import *

time_now = time.strftime("%Y%m%d-%H%M", time.localtime())

def clean_json_string(s):
    s = re.sub(r'[\x00-\x1F\x7F]', '', s)
    return s

def initdate(path,testmodel):
    data=pd.read_excel(path)
    biasname=data['biasname']
    des=data['description']
    case=data['case']
    criterion=data['criterion']
    ans=data[testmodel]
    solved_data=[]
    for i in range(len(data[testmodel])):
        tmp=('The cognitive biases currently detected is:\n'+str(biasname[i])+
             '\nThe description of this cognitive bias is:\n'+str(des[i])+
             '\nscenario question is:\n'+str(case[i])+
             '\nEvaluation Criteria is:\n'+str(criterion[i])+
             '\noutput of a large model is:'+str(ans[i]))+test_prompt
        solved_data.append(tmp)
    return solved_data

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--data_path',default='./Data/test_all.xlsx',type=str)
    parser.add_argument('--test_model',default='llama3-70B',type=str)
    parser.add_argument('--used_model',default='gpt-4-turbo',type=str)
    parser.add_argument('--start_index',default=0,type=int)
    args=parser.parse_args()

    orignin_data=pd.read_excel(args.data_path)['case']
    solved_data=initdate(args.data_path,testmodel=args.test_model)
    res=[]
    reason=[]
    case=[]
    LLM = OpenAI_chat('./Key_GPT_0.txt',"https://api.openai.com/v1")
    for i in tqdm(range(args.start_index,len(solved_data))):
        case.append(orignin_data[i])
        if solved_data[i]=='error':
            res.append(1)
            reason.append('error')
        else:    
            test_message=[
              {"role": "system", "content": question_prompt},
              {"role": "user", "content": solved_data[i]}
            ]
            while True:
                try:
                      response=LLM.get_LLM_message(used_model=args.used_model, message=test_message,temperature=0)
                      response=clean_json_string(response)
                      response = json.loads(response)
                      break
                except:
                      print(f'number {i+1} case error.')
                continue
            res.append(int(response[0]['Result']))
            reason.append(response[0]['Reason'])

        resdata=pd.DataFrame({'case':case,"res":res,'reason':reason})
        resdata.to_excel(f'./res/{args.test_model}/{args.test_model}_eval_{time_now}.xlsx')