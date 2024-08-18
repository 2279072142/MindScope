from datasets import Dataset
import pandas as pd
def data_presolver():
    testdata=pd.read_excel('../Data/Testdataset/test_all.xlsx')
    evaldata=pd.read_excel('../Data/static_data/eval_all.xlsx')
    input=testdata['case']
    llms=['GPT-4','Chatglm-6B','llama-7B','llama-13B','llama-70B','GPT-3.5-turbo','Vicuna-13B','Vicuna-7B','Vicuna-33B','Gemini Pro']

    id=0
    ids=[]
    llm=[]
    inputs=[]
    outputs=[]
    isbias=[]
    checkbias=[]

    for i in range(len(llms)):
        curcheckbias=testdata['biasname']
        curoutput=testdata[llms[i]]
        cureval=evaldata[llms[i]]
        for j in range(len(input)):
            ids.append(id)
            checkbias.append(curcheckbias[j])
            llm.append(llms[i])
            inputs.append((input[j]))
            outputs.append(curoutput[j])
            isbias.append(cureval[j])
            id+=1

    res=pd.DataFrame({'ids':ids,'checkbias':checkbias,'input':inputs,'output':outputs,'llm':llm,'isbias':isbias})
    res.to_excel('./MindScope_result.xlsx')
def data_postsolver():
    data=pd.read_excel('MindScope_result.xlsx')
    dataset_from_json=Dataset.from_json(data)
    print(dataset_from_json)

if __name__ == '__main__':
    #data_presolver()
    data_postsolver()