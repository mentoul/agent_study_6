"""
OpenAI兼容DEEKSEEK开头基础调用API
"""
import os
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()
client=OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
    )
MODEL=os.getenv("DEEPSEEK_MODEL")

response=client.chat.completions.create(
    model=MODEL,
    message=[
        {"role":"system","content":""},
        {"role":"user","content":""}
    ],
    #参数调整、限制
)
ans=response.choices[0].message.content


"""
prompt工程：zero_shot/few_shot/chain_of_thougth(cot推理)
"""
def zero_shot():
    prompt="你是XXX，简要回答"
def few_cot():
    prompt="""
样例一：

样例二：

样例三：

现在处理：

"""
def chain_of_thought():
    prompt="""
请一步一步分析：
1、
2、
3、
4、
"""

"""
输出token与格式化输出(JSON)
"""
import json
def token_json():
    response=client.chat.sompletions.create(
        response_format={"type":"json_object"}
    )
    usage=response.usage
    total_tokens=usage.total_tokens

res=token_json()
print(json.dumps(res,ensure_ascii=False))


"""
function_calling:工具调用
"""
TOOLS=[
    {
        "type":"function",
        "function":{
            "name":"",
            "description":"",
            "parameters":{
                "type":"object",
                "properties":{
                    "参数":{"type":"","description":""}
                },
                "required":["参数"],
                #不需要参数：
                "type":"object",
                "properties":{},
                "required":[],
            },
        },
    },
]

def agent_run(user_input):
    messages=[{"role":"user","content":user_input}],
    response=client.chat.completions.create(
        tools=TOOLS,
        messaages=messages,
        tool_choice="auto",
        tool_choice="none",
        tool_choice={
            "type":"function",
            "function":{
                "name":""
            }
        }
    )
    res=response.choices[0].message

    if res.tool_calls:
        for tool_call in res.tool_calls:
            name=tool_call.function.name
            args=json.loads(tool_call.function.arguments)
            messages.append({
                "role":"assistant",
                "tool_calls":[tool_call.model_dump()]
            })
            print(f"[TOOL]调用{name},参数：{args}")
            result="执行函数"()
            messages.append({
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content":result
            })
        final=client.chat.completions.create(
            model=MODEL,
            messages=messages)
        return final.choices[0].message.content
    return res.content

res=agent_run()
print(res)

"""
流式+指数退避
"""
import random
import time
from openai import OpenAI,RateLimitError, APITimeoutError, BadRequestError, APIError
def all_with_backoff():
    max_retry=4
    base_delay=1

    for attempt in range(max_retry):
        try:
            print(f"第{attempt+1}次尝试")
            response=client.chat.completions.create(
                stream=True
            )
            full_content=""
            for chunk in response:
                if chunk.choices and chunk.choices[0].delta.content:
                    delta_text=chunk.choices[0].delta.content
                    full_content+=delta_text
                    #end="":默认print会自动换行，这个代表打印完后不换行；flush=True:强制立刻把内容刷到控制台显示，不放到缓存区
                    print(delta_text,end="",flush=True) 
            print(f"请求成功，完整回答：{full_content}")

        except (RateLimitError, APITimeoutError, APIError) as e:
                if attempt == max_retry - 1:
                    print(f"\n已经达到最大重试次数，失败:{e}")
                    return None
                wait_time = base_delay * (2 ** attempt)
                jitter = random.uniform(0, 1)  #随机抖动
                total_wait = wait_time + jitter
                print(f"故障:{type(e).__name__},等待{total_wait:.2f} s 之后重试")
                time.sleep(total_wait)

        except BadRequestError as e:
            err_payload = e.response.json()
            err_code = err_payload["error"]["code"]
            if err_code == "context_length_exceeded":
                print("上下文 token 超限，不需要重试！")
                return None
            else:
                print(f"非法请求错误：{e}")
                return None
           
all_with_backoff()

"""
流式下的token输出
"""
response=client.chat.completions.create(
    stream_opention={"include_usage":True}
)
usage=None
for chunk in response:
    if getattr(chunk,"usage",None):
        usage=chunk.usage
    if usage:
        print(f"token总消耗：{usage.total_token}")

"""
function_calling的流式和指数退避
"""
import random
import time
from openai import OpenAI,RateLimitError, APITimeoutError, BadRequestError, APIError

class MsgMock:
    def __init__(self,content,tool_calls):
        self.tool_calls=tool_calls
        self.content=content

TOOLS=[
    {
        "type":"function",
        "function":{
            "name":"",
            "description":"",
            "parameters":{
                "type":"object",
                "properties":{
                    "参数":{"type":"","description":""}
                },
                "required":["参数"],
                #不需要参数：
                "type":"object",
                "properties":{},
                "required":[],
            },
        },
    },
]


def llm_stream(messages, tools=None):
    max_retry=4
    base_delay=1
    for attempt in range(max_retry):
        try:
            print(f"发起第{attempt+1}次请求")
            stream=client.chat.completions.create(
                tool_choice="auto",
                stream=True,
            )
            full_content="" 
            tool_calls_collect=[]
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta:
                    delta=chunk.choices[0].delta
                    if delta.tool_calls:
                        for tc in delta.tool_calls:
                            if tc.index >= len(tool_calls_collect):
                                tool_calls_collect.append({
                                    "id":tc.id,
                                    "type":"function",
                                    "function":{
                                        "name":"",
                                        "argumenets":""
                                    }
                                })
                            tool_calls_collect[tc.index]["function"]["name"]+=tc.function.name or ""
                            tool_calls_collect[tc.index]["function"]["arguments"]+=tc.function.arguments or ""
                    elif delta.content:
                        delta_text=delta.content
                        full_content+=delta_text
                        print(delta_text,end="",flush=True)
            print()
            if tool_calls_collect:
                tool_calls=tool_calls_collect
            else:
                tool_calls=None
            mock_res_msg=MsgMock(full_content,tool_calls)
            return mock_res_msg


        except (RateLimitError, APITimeoutError, APIError) as e:
            if attempt == max_retry -1:
                print("多次重试失败", e)
                return None
            wait_time = base_delay * (2**attempt)
            jitter = random.uniform(0,1)
            total_wait = wait_time + jitter
            print(f"故障:{type(e).__name__},等待{total_wait:.2f}s重试")
            time.sleep(total_wait)
        except BadRequestError as e:
            print(f"参数错误：{e}")
            return None           
        
def agent_run(user_input):
    messages=[{"role":"user","content":user_input}],
    res=llm_stream(messages, tools=TOOLS)
    if res is None:
        return "请求失败"   
    if res.tool_calls:
        for tool_call in res.tool_calls:
            name=tool_call["function"]["name"]
            args=json.loads(tool_call["function"]["arguments"])
            messages.append({
                "role":"assistant",
                "tool_calls":[tool_call]
            })
            print(f"[TOOL]调用{name},参数：{args}")
            result="执行函数"()
            messages.append({
                "role":"tool",
                "tool_call_id":tool_call.id,
                "content":result
            })
        final=llm_stream(
            model=MODEL,
            messages=messages)
        return final.content
    return res.content

res=agent_run()
print(res)