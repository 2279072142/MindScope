import openai
from openai import OpenAI

class OpenAI_chat():
    def __init__(self,key_path,url_base):
        self.key_path=key_path
        self.url_base=url_base
        self.client=OpenAI(api_key=open(self.key_path, 'r').read(),base_url=url_base)

    def get_LLM_message(self, message, used_model='gpt-4-turbo',method='api',temperature=0.1):
        if method == 'api':
            while True:
                try:  
                    response = self.client.chat.completions.create(
                        model=used_model,
                        messages=message,
                        temperature=temperature,
                        presence_penalty=1.0,
                        frequency_penalty=1.0,
                    
                    )
                    return response.choices[0].message.content
                except Exception as e:
                    print('api--' + self.key_path + ',did not work.--------- {}'.format(e))
                    time.sleep(15)