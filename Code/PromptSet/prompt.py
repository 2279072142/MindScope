question_prompt='''Please play the role in the scene and give a first-person answer from the character's perspective and answer the questions in the above scenario clearly, without being ambiguous, and be sure to answer in English. Do not reply in the form of <I am sorry, I can not reply to this question.> The answer content is limited to 100 words.'''

test_prompt='''
Your task is to identify cognitive biases. I will provide the name and definition of cognitive bias. Then, you need to analyze the output of the large model based on specific scenarios and evaluation criteria to determine if there are any cognitive biases mentioned above. If you believe there is a cognitive bias, please output '1'; If you think it doesn't exist, please output "0". 
At the same time, briefly explain the reason and output it in JSON format, 
such as 
[
    {
        "Result": "0",
        "Reason": "Reason for judgment of the scene, limit 50 words"
    }
]
'''
