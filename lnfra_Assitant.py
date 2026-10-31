import os
import json
import time
import random
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError, APITimeoutError, BadRequestError, APIError

load_dotenv()
client=OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=os.getenv("DEEPSEEK_BASE_URL")
)
MODEL=os.getenv("DEEPSEEK_MODEL")
SESSION_FILE="chat_session.json"

"""
功能实现计划：
1、openai兼容deepseek框架
2、3个TOOLS
3、工具调用
4、流式+指数退避
5、多轮对话
6、对话状态保持
"""
class MsgMock:   #新建一个类，用来模拟SDK返回的message对象
    """
    __init__:构造方法，创建这个类的实例的时候自动执行，
    两个形参content,tool_calls,创建对象的时候传这两个值进来
        
    """
    def __init__(self, content, tool_calls):   
        self.tool_calls = tool_calls
        self.content = content


TOOLS=[
    {
        "type":"function",
        "function":{
            "name":"check_mysql_status",
            "description":"检查MYSQL实例的运行状态，包括连接数和慢查询",
            "parameters":{
                "type":"object",
                "properties":{
                    "host":{"type":"string","description":"MYSQL实例IP或主机名"}
                },
                "required":["host"],
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"check_disk_usage",
            "description":"检查指定路径的磁盘使用率",
            "parameters":{
                "type":"object",
                "properties":{
                    "path":{"type":"string","description":"要检查的路径，如/data"}
                },
                "required":["path"],
            },
        },
    },
    {
        "type":"function",
        "function":{
            "name":"check_Nginx_health",
            "description":"检查Nginx健康状态，包括活跃连接数和QPS",
            "parameters":{
                "type":"object",
                "properties":{},
                "required":[],
            },
        },
    },
]
#模拟执行
def execute_tool(name,args):
    if name=="check_mysql_status":
        return f"MYSQL{args['host']}:连接数150/200，慢查询3条，运行正常"
    elif name=="check_disk_usage":
        return f"路径{args['path']}的磁盘使用率：78%"
    elif name=="check_Nginx_health":
        return "Nginx:active-45,requests/sec=320,200=99.2%,5xx=0.1%"
    return "未知工具"



# 新增：流式+指数退避，返回拼接好的message对象信息，兼容你原来写法
#如果你不传tools 这个参数,tools就自动等于None
def llm_stream_req(messages, tools=None, tool_choice="auto"):  #
    max_retry = 4
    base_delay = 1
    for attempt in range(max_retry):
        try:
            print(f"\n===== 发起请求，第 {attempt + 1} 次尝试 =====")
            stream = client.chat.completions.create(
                model=MODEL,
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                stream=True,
                stream_options={"include_usage": True}
            )
            full_content = ""  #字符串容器，专门用来拼接普通文字回答
            tool_calls_collect = []  #列表容器，专门存放要调用的工具，一个元素代表一次工具调用
            for chunk in stream:  #循环读取每一块chunk
                if chunk.choices and chunk.choices[0].delta:   #chunk.choices:判断这块分片有没有返回内容、.delta:流式专属字段
                    delta = chunk.choices[0].delta  
                    #delta=当前这一小块新增的数据，二选一：工具调用/普通文字
                    if delta.tool_calls:  #如果小块delta里面是工具调用信息，说明模型打算调用工具了
                        """
                        拿到一块工具分片 tc，先看：我们列表里有没有对应编号的容器。
                        没有就新建一个空容器；有就直接把当前分片的内容追加拼到这个容器里面。
                        """
                        for tc in delta.tool_calls:   #tc:tool_call的碎片  delta.tool_calls 是这一个分片里面的工具片段，遍历它，tc 就是当前这个工具分片
                            if tc.index >= len(tool_calls_collect):  #>=说明这是一个全新工具，还没有容器；tc.index:模型给的编号，代表当前这个分片属于第几个工具调用，从0开始
                                tool_calls_collect.append({
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": "",    #用来慢慢拼接工具名
                                        "arguments": ""}   #用来拼接参数字符串
                                })
                            """
                            通过tc.index找到对应容器
                            ["function"]["name"]：进入盒子里面的function，
                            找到name这个字符串，它用来存工具名字，如check_Nginx_health
                            API 返回的分片里，有些 chunk 的tc.function.name是None(没有内容)
                            如果tc.function.name有文字，用它本身，如果是None,替换成空字符串""
                            ["function"]["arguments"]：参数字符串
                            """
                            tool_calls_collect[tc.index]["function"]["name"] += tc.function.name or ""
                            tool_calls_collect[tc.index]["function"]["arguments"] += tc.function.arguments or ""
                   
                    elif delta.content:
                        #如果没有工具调用信息
                        delta_text = delta.content
                        full_content += delta_text
                        print(delta_text, end="", flush=True)
            print()  #不带任何参数 = **只输出一个换行符 \n**
            
            if tool_calls_collect:
                tool_calls = tool_calls_collect
            else:
                tool_calls = None
            mock_res_msg = MsgMock(full_content, tool_calls)
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


def agent_run(user_input,messages,tool_choice="auto"):
    messages.append({"role":"user","content":user_input}) 

    while True:
        res = llm_stream_req(messages, tools=TOOLS, tool_choice=tool_choice)
        if res is None:
            return "请求失败"
        if res.tool_calls:
            for tool_call in res.tool_calls:
                name=tool_call["function"]["name"] #取出工具名字
                args=json.loads(tool_call["function"]["arguments"])  #把json字符串参数转为python字典
                #把大模型想调用工具这件事存入对话历史           
                messages.append({
                    "role":"assistant",
                    "tool_calls":[tool_call]
                })
                print(f"[TOOL]调用{name},参数；{args}")
                result=execute_tool(name,args)
                messages.append({
                    "role":"tool",
                    "tool_call_id":tool_call["id"],
                    "content":result
                })
            continue
        else:
            break
    return True

def save_session(messages):
    with open(SESSION_FILE,"w",encoding="utf-8") as f:   #自动打开写入文件，用完关闭
        json.dump(messages,f,ensure_ascii=False,indent=2)

def load_session():
    if os.path.exists(SESSION_FILE):  #判断是否有文件
        with open(SESSION_FILE,"r",encoding="utf-8") as f:
            return json.load(f)
    return []

if __name__=="__main__":
    print("========Infra运维助手、命令行交互========")
    print("指令：exit退出并保存|load加载会话|clear清空对话")
    messages=[]
    while True:
        user_input=input("\n你：")
        cmd=user_input.strip().lower()  #strip:删掉前后空格、换行；lower：全部转为小写
        if cmd=="exit":
            save_session(messages)
            print("会话已保存！退出")
            break
        elif cmd=="load":
            messages=load_session()
            print(f"加载成功，当前上下文长度：{len(messages)}")
            continue
        elif cmd=="clear":
            messages=[]
            print("对话已经清除！")
            continue
        print("Agent:",end="")
        ok=agent_run(user_input,messages)
        if not ok:
            print("请求失败！")
        save_session(messages)

