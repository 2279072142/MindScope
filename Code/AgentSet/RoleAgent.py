import openai
import time
import utils.general_utils as utils
import utils.token_counting as token_helper
from openai import OpenAI
#import CGMI_main as all
from random import  randint
from LLM.azure_openai import Azure
import os
class RoleAgent:
    def __init__(self, name, used_model, chat_log, key_path, WM_num):
        self.name = name

        self.system=''
        self.key_path = key_path  
        self.WM_num = WM_num  
        self.used_model = used_model 
        self.ChatLog = chat_log  
        self.WorkingMemory = []  
        self.LongMemory = []  
        self.ShortMemory = [] 
        # self.azure = Azure('xx',
        #             'xx',
        #             './Token/access_token.txt', './Token/refresh_token.txt')
    def set_system(self, content):
        self.system={"role": "system", "content": content}
        self.ShortMemory.append(content)
        #self.WorkingMemory.append({"role": "system", "content": content})
        self.ChatLog.write('---RoleSet for ' + self.name + '：\n' + content + '\n\n')

    def receive_message(self, content):
        self.ShortMemory.append(content)
        self.ChatLog.write('---Get from others ：\n' + content + '\n\n')

    def update_ShortMemory(self, content):
        self.ShortMemory.append(content)
        self.ChatLog.write('---Update ShortMemory ：\n' + content + '\n\n')
    def response_LLMs_gpt4(self,message):
        keyindex = randint(0, 3)
        if keyindex == 0:
            self.key_path = './Key_GPT_0.txt'
        elif keyindex == 1:
            self.key_path = './Key_GPT_1.txt'
        elif keyindex == 2:
            self.key_path = './Key_GPT_2.txt'
        while True:
            try:  
                # print(openai.api_key)
                input_token = token_helper.count_message_tokens(message, self.used_model)
                openai.api_key = open(self.key_path, 'r').read()
                response = openai.ChatCompletion.create(
                    model='gpt-4',
                    messages=message,
                    temperature=0.2,
                    presence_penalty=1.0,
                    frequency_penalty=1.0
                )
                response = response['choices'][0]['message'].content
                ouput_token = token_helper.count_string_tokens(response, self.used_model)
                curcost = token_helper.count_dollar(input_token, ouput_token, self.used_model)
                all.allcost += curcost
                return response

            except Exception as e:
                print('api--' + self.key_path + ',did not work.--------- {}'.format(e))
                time.sleep(15)
                if '0' in self.key_path:
                    self.key_path = self.key_path.replace('0', '1')
                elif '1' in self.key_path:
                    self.key_path = self.key_path.replace('1', '2')
                elif '2' in self.key_path:
                    self.key_path = self.key_path.replace('2', '0')
    def response_LLMs(self, message,method='api',log=None):
        if method == 'api':
            # os.environ["http_proxy"] = "127.0.0.1:7890"
            # os.environ["https_proxy"] = "127.0.0.1:7890"
            keyindex=randint(0,3)
            if keyindex==0:
                self.key_path='./Key_GPT_0.txt'
            elif keyindex==1:
                self.key_path='./Key_GPT_1.txt'
            elif keyindex==2:
                self.key_path='./Key_GPT_2.txt'
            while True:
                try: 

                    api_key=open(self.key_path,'r').read()
                    client=OpenAI(api_key=api_key,base_url='https://api.openai.com/v1')
                    response =client.chat.completions.create(
                        model=self.used_model,
                        messages=message,
                        temperature=0.1,
                        presence_penalty=1.0,
                        frequency_penalty=1.0
                    )
                    if log is not None:
                        log.write(self.name + 'said:' + response + '\n')
                    return response.choices[0].message.content

                except Exception as e:
                    print('api--' + self.key_path + ',did not work.--------- {}'.format(e))
                    time.sleep(15)
                    if '0' in self.key_path:
                        self.key_path = self.key_path.replace('0', '1')
                    elif '1' in self.key_path:
                        self.key_path = self.key_path.replace('1', '2')
                    elif '2' in self.key_path:
                        self.key_path = self.key_path.replace('2', '0')


    def chat(self, request):
        if(len(self.ShortMemory)>self.WM_num+1):
            self.summarize()
        memory='This is your memory:'
        for i in range(len(self.LongMemory)):
            memory = self.LongMemory[i] + '\n'
            # if i==0:
            #     memory='This is your long memory:'+self.LongMemory[i]+'\n'
            # else:
        for i in range(len(self.ShortMemory)):
            memory=memory+self.ShortMemory[i]+'\n'
        prompt=[self.system,{'role':'user','content':memory+'\n Based on your memory, answer the following questions:'+request}]
        response = self.response_LLMs(prompt)
        self.ShortMemory.append(request)
        print("################################")
        print(self.name+"说:"+response)
        print("################################")
        self.ChatLog.write('---' + self.name + '：\n' + response + '\n\n')
        return response

    def del_message(self, idx):
        del self.ShortMemory[idx]

    def summarize(self):
  
        self.history_memory = str(self.ShortMemory)  

        summary_system = 'The cognitive structure includes the following memory modules:\n' \
                         '1. Short Term Memory Module (STM): The main function of this module is to temporarily store information from perceptual input and long-term memory for use in current cognitive tasks. Short term memory usually has limited capacity and duration, and information gradually disappears.\n' \
                         '2. Long Term Memory Module (LTM): A long-term memory module is used to store an individual‘s knowledge, experience, and skills, which can be retained for a relatively long period of time and can be retrieved and applied at any time. The long-term memory module is further subdivided into:\n' \
                         '2.1. Memory of other people’s actions: including conversations they have with themselves, saved in the format of "xxx once said: xxx"\n' \
                         '2.2. Personal memory: including actions initiated by oneself and the content of conversations initiated by oneself, preserved in the format of "I once said to xxx: xxxx"\n'



        pm_prompt = "The following content is" + self.name + 'Please summarize the short-term memory to form "long-term memory". The specific content is as follows:\n'
        summary_message_temp = [
            {"role": "system", "content": summary_system},
            {"role": "user", "content": pm_prompt + self.history_memory}
        ]
        summary_memory = self.response_LLMs(summary_message_temp)
        if(len(self.ShortMemory)>self.WM_num+1):
            self.ShortMemory = self.ShortMemory[:1] + self.ShortMemory[-self.WM_num:]
        self.LongMemory.append(summary_memory)
        self.ShortMemory = []