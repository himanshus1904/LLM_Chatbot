import gradio as gr
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_name):
    global model, tokenizer
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return f"Model {model_name} loaded successfully!"


def chatbot_interface(input_text, history):
    history.append(input_text)
    history_string = "\n".join(history)

    inputs = tokenizer.encode_plus(history_string, return_tensors="pt", add_special_tokens=True)
    outputs = model.generate(**inputs)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    history.append(response)
    return response, history


with gr.Blocks() as demo:
    gr.Markdown(
        "### Enter a model name to load. You can refer to the [Hugging Face Models](https://huggingface.co/models?pipeline_tag=text2text-generation&sort=trending) for model names.")

    model_name_input = gr.Textbox(label="Model Name", placeholder="e.g., facebook/blenderbot-400M-distill")
    load_button = gr.Button("Load Model")
    load_output = gr.Textbox(label="Output")

    with gr.Tab("Chatbot"):
        chatbot_input = gr.Textbox(label="Input")
        chatbot_output = gr.Textbox(label="Response")
        conversation_history = gr.State([])

    load_button.click(load_model, inputs=model_name_input, outputs=load_output)
    chatbot_input.submit(chatbot_interface, inputs=[chatbot_input, conversation_history],
                         outputs=[chatbot_output, conversation_history])

demo.launch()