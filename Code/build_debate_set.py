import json
import pandas as pd
from AgentSet.UniversalAgent import *
import ast
from PromptSet.Prompts_library import *
from tqdm import tqdm
from random import randint


# def solve():
#     test_data=pd.read_excel('./Data/test_all.xlsx')
#     eval_data=pd.read_excel('./Data/eval_all.xlsx')
#     case=test_data['case']
#     biasname=test_data['biasname']
#     llm_name=['gpt-4','gpt-3.5','llama-7b','llama-13b','llama-70b','chatglm-6b','vicuna-7b','vicuna-13b','vicuna-33b','gemini']
#     res=[]
#     for name in llm_name:
#         for i in range(len(case)):
#             res.append(
#                 {'case':case[i],
#                  'biasname':biasname[i],
#                  'exit':int(eval_data[name][i]),
#                  'response':test_data[name][i],
#                  'origin':name}
#             )
#     with open('./Data/static_dataset.json','w') as file:
#         json.dump(res,file,indent=4)
def get_onesingleagent_res():
    data = pd.read_excel('./Data/res_ontsingleagent.xlsx')
    biasname = data['biasname']
    res = data['res']
    label = data['label']
    corret = []
    cnt = 0
    for i in range(len(biasname)):
        if label[i] == 1:
            if (biasname[i].lower() in res[i].lower()):
                cnt += 1
                corret.append(1)
            else:
                corret.append(0)
        else:
            if 'no bias' in res[i].lower():
                cnt += 1
                corret.append(1)
            else:
                corret.append(0)
    print(cnt / len(biasname))
    pd.DataFrame({'label': corret}).to_excel('./label_onesingleagent.xlsx')


def clean_json_string(s):
    # Remove control characters
    s = re.sub(r'[\x00-\x1F\x7F]', '', s)
    # Fix common JSON string issues, if necessary, here

    pattern = r'\{.*\:\s*\{.*\:.*\}\}'

    match = re.search(pattern, s)

    if match:
        json_str = match.group()

        json_data = json.loads(json_str)
        return json_data
    else:
        return ""



def solve(text):

    matched = re.search(r"\[(.*?)\]", text)

    extracted_text = matched.group(0) if matched else ""
    return extracted_text


def gethouxuanset():
    data = pd.read_excel('./Data/res_test25_1.xlsx')
    res = data['res']
    biasname = data['biasname']
    candicate = []
    for i in range(len(res)):
        cands = f'["{biasname[i].lower()}",'
        try:
            cur = ast.literal_eval(solve(res[i]))
        except:
            print(f'?{i}???????')
            candicate.append(cands)
            continue
        for j in range(len(cur)):
            if cur[j].lower() == biasname[i].lower():
                continue
            else:
                cands += f'"{cur[j].lower()}"]'
                break
        candicate.append(cands)
    pd.DataFrame({'candicate': candicate}).to_excel('./candicate.xlsx')


def build_train_set():
    data = pd.read_excel("./Data/never_used_data/testset25.xlsx")
    biasname = data['biasname']
    case = data['case']
    candicate = data['candicate']
    lflags = []
    rflags = []
    resl = []
    resr = []
    records = []
    with open('./Data/CB_Object.json', 'r') as file:
        CB_Object = json.load(file)
    allcost = 0
    for j in tqdm(range(0, len(case))):
        text = case[j]
        curcan = ast.literal_eval(solve(candicate[j]))
        elements = []
        for i in range(len(curcan)):
            for item in CB_Object:
                if (item['bias_name'].lower() == curcan[i].lower()):
                    curcan.append(curcan[i].lower())
                    elements.append(item['elements'])
        if len(elements) != 2:
            for item in CB_Object:
                if (item['bias_name'].lower() == "confirmation bias"):
                    curcan.append(curcan[i].lower())
                    elements.append(item['elements'])
            curcan.append("confirmation bias")
        l = randint(0, 1)
        if l == 0:
            r = 1
        else:
            r = 0
        l_agent = checkBiasAgent(name=names[l], bias=curcan[l], object=elements[l],
                                 model='gpt-3.5-turbo-16k')
        r_agent = checkBiasAgent(name=names[r], bias=curcan[r], object=elements[r],
                                 model='gpt-3.5-turbo-16k')
        allrecord = ""
        # ??
        l_request = f"you think {l_agent.bias} exists in the current scene. Please demonstrate why {l_agent.bias} exists in {text}. The attribute that requires special attention for this cognitive bias is {l_agent.object}. The limit is 80 words. "
        l_response = l_agent.chat(l_request)
        r_agent.receive_message(l_response)
        # print(f'{l_agent.name} said:{l_response}\n')
        r_request = f"you think {r_agent.bias} exists in the current scene. Please demonstrate why {r_agent.bias} exists in {text}. The attribute that requires special attention for this cognitive bias is {r_agent.object}, and the limit is 80 words. "
        r_response = r_agent.chat(r_request)
        l_agent.receive_message(r_response)
        # print(f'{r_agent.name} said:{r_response}\n')
        allrecord += f'{l_agent.name} said:{l_response}\n'
        allrecord += f'{r_agent.name} said:{r_response}\n'
        # ??
        l_request = f"You think {l_agent.bias} exists in the current scene, and the other party thinks there is {r_agent.bias}, please refute the other party's point of view, limit 100 words"
        l_response = l_agent.chat(l_request)
        r_agent.receive_message(l_response)
        # print(f'{l_agent.name} said:{l_response}\n')
        r_request = f"You think {r_agent.bias} exists in the current scene, and the other party thinks there is {l_agent.bias}, please refute the other party's point of view, limit 100 words"
        r_response = r_agent.chat(r_request)
        l_agent.receive_message(r_response)
        # print(f'{r_agent.name} said:{r_response}\n')
        allrecord += f'{l_agent.name} said:{l_response}\n'
        allrecord += f'{r_agent.name} said:{r_response}\n'
        # ??
        l_request = f"Based on the debate process between you and your opponent, please summarize the reasons why you think {l_agent.bias} exists in the current scene {text}. Limit 100 words."
        l_response = l_agent.chat(l_request)
        r_agent.receive_message(l_response)
        # print(f'{l_agent.name} said:{l_response}\n')
        r_request = f"Based on the debate process between you and your opponent, please summarize the reasons why you think {r_agent.bias} exists in the current scene {text}. Limit 100 words."
        r_response = r_agent.chat(r_request)
        l_agent.receive_message(r_response)
        # print(f'{r_agent.name} said:{r_response}\n')
        allrecord += f'{l_agent.name} said:{l_response}\n'
        allrecord += f'{r_agent.name} said:{r_response}\n'
        score_agent1 = checkBiasAgent(model='gpt-3.5-turbo-16k',personality=judge_role1)
        score_agent2 = checkBiasAgent(model='gpt-3.5-turbo-16k',personality=judge_role2)
        # score,flag=score_agent.scoring(allrecord,l_agent.name,r_agent.name,l_agent.bias,r_agent.bias,text)

        lscore_tmp, lflag = score_agent1.scoring_one(allrecord, l_agent.name, l_agent.bias, text,temperature=0.3)
        rscore, rflag = score_agent2.scoring_one(allrecord, r_agent.name, r_agent.bias, text,temperature=0.5)
        rscore_tmp, rflag = score_agent1.scoring_one(allrecord, r_agent.name, r_agent.bias, text,temperature=0.3)
        lscore, lflag = score_agent2.scoring_one(allrecord, l_agent.name, l_agent.bias, text,temperature=0.5)
        l_agent.name=l_agent.name.lower()
        r_agent.name=r_agent.name.lower()
        lscore[l_agent.name]['Argument Support'] = (lscore_tmp[l_agent.name]['Argument Support'] + lscore[l_agent.name][
            'Argument Support']) / 2
        lscore[l_agent.name]['Logical Consistency'] = (lscore_tmp[l_agent.name]['Logical Consistency'] +
                                                       lscore[l_agent.name]['Logical Consistency']) / 2
        lscore[l_agent.name]['Refutation Effectiveness'] = (lscore_tmp[l_agent.name]['Refutation Effectiveness'] +
                                                            lscore[l_agent.name]['Refutation Effectiveness']) / 2
        lscore[l_agent.name]['Argument Completeness'] = (lscore_tmp[l_agent.name]['Argument Completeness'] +
                                                         lscore[l_agent.name]['Argument Completeness']) / 2
        lscore[l_agent.name]['Persuasiveness'] = (lscore_tmp[l_agent.name]['Persuasiveness'] + lscore[l_agent.name][
            'Persuasiveness']) / 2
        lscore[l_agent.name]['Reasonability assessment of cognitive bias'] = (lscore_tmp[l_agent.name][
                                                                                  'Reasonability assessment of cognitive bias'] +
                                                                              lscore[l_agent.name][
                                                                                  'Reasonability assessment of cognitive bias']) / 2

        rscore[r_agent.name]['Argument Support'] = (rscore_tmp[r_agent.name]['Argument Support'] + rscore[r_agent.name][
            'Argument Support']) / 2
        rscore[r_agent.name]['Logical Consistency'] = (rscore_tmp[r_agent.name]['Logical Consistency'] +
                                                       rscore[r_agent.name]['Logical Consistency']) / 2
        rscore[r_agent.name]['Refutation Effectiveness'] = (rscore_tmp[r_agent.name]['Refutation Effectiveness'] +
                                                            rscore[r_agent.name]['Refutation Effectiveness']) / 2
        rscore[r_agent.name]['Argument Completeness'] = (rscore_tmp[r_agent.name]['Argument Completeness'] +
                                                         rscore[r_agent.name]['Argument Completeness']) / 2
        rscore[r_agent.name]['Persuasiveness'] = (rscore_tmp[r_agent.name]['Persuasiveness'] + rscore[r_agent.name][
            'Persuasiveness']) / 2
        rscore[r_agent.name]['Reasonability assessment of cognitive bias'] = (rscore_tmp[r_agent.name][
                                                                                  'Reasonability assessment of cognitive bias'] +
                                                                              rscore[r_agent.name][
                                                                                  'Reasonability assessment of cognitive bias']) / 2

        print(lscore)
        print(rscore)
        lflags.append(lflag)
        rflags.append(rflag)
        resl.append(lscore)
        resr.append(rscore)
        records.append(allrecord)
        pd.DataFrame({"resl": resl, "resr": resr, "record": records, "lflag": lflags, "rflag": rflags}).to_excel(
            "./debate_record4.xlsx")
    print(f"all cost {allcost}")
    print("test compete!!!")


if __name__ == '__main__':
    build_train_set()