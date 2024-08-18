from AgentSet.UniversalAgent import *
from AgentSet.RoleAgent import *
PROMPT_TEMPLATE = """
NOTICE
Role: You are a professional engineer; the main goal is to write PEP8 compliant, elegant, modular, easy to read and maintain Python 3.9 code (but you can also use other programming language)
ATTENTION: Use '##' to SPLIT SECTIONS, not '#'. Output format carefully referenced "Format example".

## Code: {filename} Write code with triple quoto, based on the following list and context.
1. Do your best to implement THIS ONLY ONE FILE. ONLY USE EXISTING API. IF NO API, IMPLEMENT IT.
2. Requirement: Based on the context, implement one following code file, note to return only in code form, your code will be part of the entire project, so please implement complete, reliable, reusable code snippets
3. Attention1: If there is any setting, ALWAYS SET A DEFAULT VALUE, ALWAYS USE STRONG TYPE AND EXPLICIT VARIABLE.
4. Attention2: YOU MUST FOLLOW "Data structures and interface definitions". DONT CHANGE ANY DESIGN.
5. Think before writing: What should be implemented and provided in this document?
6. CAREFULLY CHECK THAT YOU DONT MISS ANY NECESSARY CLASS/FUNCTION IN THIS FILE.
7. Do not use public member functions that do not exist in your design.

-----
# Context
{context}
-----
## Format example
-----
## Code: {filename}
```python
## {filename}
...
```
-----
"""
code_log = open('../code_log.txt', 'a+', encoding='utf-8')  # 创建进展监督的日志文件
code_generator=BaseAgent("gpt-3.5-turbo-16k", "../LLM/Openaikey/Key_GPT_0.txt", code_log)
code_generator.system=PROMPT_TEMPLATE
code_generator.user="请生成快速排序的代码"
code_generator.setMessage()
res=code_generator.response_LLMs()
print(res)
