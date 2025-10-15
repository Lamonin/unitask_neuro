import os
import requests
import tiktoken
from langchain.docstore.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from openai import OpenAI


class GPT:
    def __init__(self):
        self.model = "openai/gpt-5-nano"
        self.log = ""
        self.search_index = None
        self.client = OpenAI(
            api_key=os.environ["OPENAI_API_KEY"], base_url=os.environ["OPENAI_URL"]
        )

    def load_search_indexes(self, url: str):
        response = requests.get(url)
        response.raise_for_status()
        return self.create_embedding(response.text)

    def num_tokens_from_string(self, string: str) -> int:
        try:
            encoding = tiktoken.encoding_for_model(self.model)
        except KeyError:
            encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(string))

    def create_embedding(self, data: str):
        splitter = CharacterTextSplitter(
            separator="\n", chunk_size=1024, chunk_overlap=0
        )
        docs = [Document(page_content=chunk) for chunk in splitter.split_text(data)]

        token_count = self.num_tokens_from_string(
            " ".join(d.page_content for d in docs)
        )
        self.log += f"Токенов в документе: {token_count}\n"

        self.search_index = Chroma.from_documents(docs, OpenAIEmbeddings())
        self.log += "Документ загружен в векторную базу.\n"
        return self.search_index

    def num_tokens_from_messages(self, messages):
        encoding = tiktoken.encoding_for_model(self.model)
        tokens_per_message, tokens_per_name = 3, 1
        num_tokens = 0
        for msg in messages:
            num_tokens += tokens_per_message
            for k, v in msg.items():
                num_tokens += len(encoding.encode(v))
                if k == "name":
                    num_tokens += tokens_per_name
        return num_tokens + 3

    def answer_index(self, system: str, topic: str, temp: float = 1.0):
        if not self.search_index:
            self.log += "Модель не обучена.\n"
            return ""

        docs = self.search_index.similarity_search(topic, k=5)
        message_content = "\n".join(
            [f"Фрагмент №{i + 1}:\n{doc.page_content}" for i, doc in enumerate(docs)]
        )

        messages = [
            {"role": "system", "content": system + "\n" + message_content},
            {"role": "user", "content": topic},
        ]

        self.log += f"Токенов в вопросе: {self.num_tokens_from_messages(messages)}\n"

        completion = self.client.chat.completions.create(
            model=self.model, messages=messages, temperature=temp
        )

        usage = completion.usage
        self.log += (
            f"Токенов вопроса: {usage.prompt_tokens}, всего: {usage.total_tokens}\n"
        )

        return completion.choices[0].message.content
