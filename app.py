# 📦 Install required packages
!pip install transformers scikit-learn pandas gradio

# ✅ Import libraries
from transformers import pipeline, set_seed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import gradio as gr

# 🎓 Sample dataset for training
data = {
    'text': [
        "The Prime Minister held a press conference today",
        "Aliens have been spotted helping with construction",
        "The stock market closed at a record high today",
        "Vampires seen at government building during night",
        "RCB own the ipl trophy"
    ],
    'label': [0, 1, 0, 1, 0]  # 0 = Real, 1 = Fake
}
df = pd.DataFrame(data)

# ✏️ Train simple classifier
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']
clf = MultinomialNB()
clf.fit(X, y)

# 🧪 Detection function
def detect_news(text):
    vec = vectorizer.transform([text])
    pred = clf.predict(vec)[0]
    return "Fake News ❌" if pred else "Real News ✅"

# ✨ Fake news generator (optional)
generator = pipeline('text-generation', model='gpt2')
set_seed(42)
def generate_fake_news(prompt):
    output = generator(prompt, max_length=100, num_return_sequences=1)
    return output[0]['generated_text']

# 🎛️ Gradio Interface
with gr.Blocks() as demo:
    gr.Markdown("## 📰 Fake News Generator & Detector")
    
    with gr.Tab("Generate Fake News"):
        prompt = gr.Textbox(label="Enter a prompt")
        output = gr.Textbox(label="Generated Fake News")
        gen_btn = gr.Button("Generate")
        gen_btn.click(generate_fake_news, inputs=prompt, outputs=output)

    with gr.Tab("Detect Fake News"):
        news = gr.Textbox(label="Enter news to verify")
        result = gr.Textbox(label="Prediction")
        detect_btn = gr.Button("Check")
        detect_btn.click(detect_news, inputs=news, outputs=result)

demo.launch()
