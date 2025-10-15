import os
import re
import tomllib

import gradio as gr

from gpt import GPT

with open("secrets.toml", "rb") as file:
    secrets = tomllib.load(file)

os.environ["OPENAI_API_KEY"] = secrets["api"]["OPENAI_API_KEY"]
os.environ["OPENAI_API_BASE"] = secrets["api"]["OPENAI_URL"]

with open("models.toml", "rb") as file:
    models = tomllib.load(file)["models"]

gpt = GPT()

with gr.Blocks(title="Нейро-сотрудники Unity") as demo:
    subject = gr.Dropdown(
        [(model["name"], i) for i, model in enumerate(models)],
        label="Нейро-сотрудник",
        value=0,
    )

    name = gr.Label(show_label=False, value=models[0]["name"])
    prompt = gr.Textbox(label="Промт", value=models[0]["prompt"], interactive=True)
    query = gr.Textbox(label="Запрос к LLM", value=models[0]["query"], interactive=True)
    link = gr.HTML(value=f"<a target='_blank' href='{models[0]['doc']}'>Документ для обучения</a>")

    def onchange(dropdown):
        model = models[dropdown]
        return [
            model['name'],
            re.sub(r"\t+|\s\s+", " ", model["prompt"]),
            model["query"],
            f"<a target='_blank' href='{model['doc']}'>Документ для обучения</a>",
        ]

    subject.change(onchange, subject, [name, prompt, query, link])

    with gr.Row():
        train_btn = gr.Button("Обучить модель")
        request_btn = gr.Button("Запрос к модели")

    def train(dropdown_index):
        gpt.load_search_indexes(models[dropdown_index]["doc"])
        return gpt.log

    def predict(p, q):
        response = gpt.answer_index(p, q)
        return [response, gpt.log]

    with gr.Row():
        response = gr.Textbox(label="Ответ LLM", lines=8)
        log = gr.Textbox(label="Лог", lines=8)

    train_btn.click(train, subject, log)
    request_btn.click(predict, [prompt, query], [response, log])

demo.launch()
