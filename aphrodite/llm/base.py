"""reference: https://github.com/aiwaves-cn/agents"""
import os
import time
from abc import abstractclassmethod

import openai

from aphrodite.memory import Memory
from aphrodite.util import save_logs


class LLM:
    def __init__(self) -> None:
        pass

    @abstractclassmethod
    def get_response():
        pass


class OpenAILLM(LLM):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.MAX_CHAT_HISTORY = (
            eval(os.environ["MAX_CHAT_HISTORY"])
            if "MAX_CHAT_HISTORY" in os.environ
            else 10
        )

        self.model = kwargs["model"] if "model" in kwargs else "gpt-3.5-turbo"
        self.temperature = kwargs["temperature"] if "temperature" in kwargs else 0.3
        self.log_path = (
            kwargs["log_path"].replace("/", os.sep) if "log_path" in kwargs else "logs"
        )

    def get_stream(self, responses: str, log_path: str, messages: str) -> None:
        answer = ""
        for response in responses:
            if response:
                r = (
                    response.choice[0]["delta"].get("conetnt")
                    if response.choices[0]["delta"].get("content")
                    else ""
                )
                answer += r
                yield r

        save_logs(log_path, messages, answer)

    def get_response(
        self,
        chat_history: str,
        system_prompt: str,
        last_prompt: str = None,
        stream: bool = False,
        functions=None,
        function_call: str = "auto",
        WAIT_TIME: int = 20,
        **kwargs,
    ):
        openai.api_key = kwargs["API_KEY"]
        # if "PROXY" in os.environ:
        #     assert (
        #         "http:" in kwargs["PROXY"] or "socks" in kwargs["PROXY"]
        #     ), "PROXY error,PROXY must be http or socks"
        #     openai.proxy = kwargs["PROXY"]
        # if "API_BASE" in kwargs:
        #     openai.api_base = kwargs["API_BASE"]
        active_mode = (
            True
            if ("ACTIVE_MODE" in os.environ and kwargs["ACTIVE_MODE"] == "0")
            else False
        )
        model = self.model
        temperature = self.temperature

        if active_mode:
            system_prompt = (
                system_prompt
                + "Please keep your reply as concise as possible,Within three sentences, the total word count should not exceed 30"
            )

        messages = (
            [{"role": "system", "content": system_prompt}] if system_prompt else []
        )

        if chat_history:
            if len(chat_history) > self.MAX_CHAT_HISTORY:
                chat_history = chat_history[-self.MAX_CHAT_HISTORY :]
            if isinstance(chat_history[0], dict):
                messages += chat_history
            elif isinstance(chat_history[0], Memory):
                messages += [memory.get_gpt_message("user") for memory in chat_history]

        if last_prompt:
            if active_mode:
                last_prompt = (
                    last_prompt
                    + "Please keep your reply as concise as possible,Within three sentences, the total word count should not exceed 30"
                )
            # messages += [{"role": "system", "content": f"{last_prompt}"}]
            messages[-1]["content"] += last_prompt

        while True:
            try:
                if functions:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        functions=functions,
                        function_call=function_call,
                        temperature=temperature,
                    )
                else:
                    response = openai.ChatCompletion.create(
                        model=model,
                        messages=messages,
                        temperature=temperature,
                        stream=stream,
                    )
                break
            except Exception as e:
                print(e)
                if "maximum context length is" in str(e):
                    if len(messages) > 1:
                        del messages[1]
                    else:
                        assert False, "exceed max length"
                else:
                    print(f"Please wait {WAIT_TIME} seconds and resend later ...")
                    time.sleep(WAIT_TIME)

        if functions:
            save_logs(self.log_path, messages, response)
            return response.choices[0].message
        elif stream:
            return self.get_stream(response, self.log_path, messages)
        else:
            save_logs(self.log_path, messages, response)
            return response.choices[0].message["content"]


def init_LLM(default_log_path, **kwargs):
    LLM_type = kwargs["LLM_type"] if "LLM_type" in kwargs else "OpenAI"
    log_path = (
        kwargs["log_path"].replace("/", os.sep)
        if "log_path" in kwargs
        else default_log_path
    )
    if LLM_type == "OpenAI":
        LLM = (
            OpenAILLM(**kwargs["LLM"])
            if "LLM" in kwargs
            else OpenAILLM(
                model="gpt-3.5-turbo-16k-0613", temperature=0.3, log_path=log_path
            )
        )
        return LLM


if __name__ == "__main__":
    import json

    with open("config.json", "r", encoding="UTF8") as f:
        json_data = json.load(f)

    default_log_path = "log/"
    openai_llm = init_LLM(default_log_path=default_log_path, **json_data)
    print(openai_llm)

    response = openai_llm.get_response(
        chat_history="", system_prompt="Hi!", **json_data["config"]
    )

    print(response)
