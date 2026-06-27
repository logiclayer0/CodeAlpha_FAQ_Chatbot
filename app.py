import tkinter as tk
from tkinter import ttk, messagebox
import nltk
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

class EnterpriseFAQBot:
    def __init__(self, window):
        self.root = window
        self.root.title("CodeAlpha AI Support Engine — FAQBot v1.0")
        self.root.geometry("600x680")
        self.root.geometry("600x680")
        self.root.configure(bg="#0d1117")
        self.root.resizable(False, False)
        
        self.faq_dataset = {
            "what is codealpha?": "CodeAlpha is a leading software development company dedicated to driving innovation across emerging technologies.",
            "what internship domains do you offer?": "We offer internships in Artificial Intelligence, Web Development, Android Development, and Cyber Security.",
            "will i get a certificate?": "Yes, you will receive a QR-verified Completion Certificate and a Unique ID Certificate upon successfully completing your tasks.",
            "what are the criteria to pass the internship?": "To be eligible for the certificate, you must complete and submit a minimum of 2 or 3 tasks within the given deadline.",
            "do you provide a letter of recommendation?": "Yes, a Letter of Recommendation (LOR) is provided to interns based on their exceptional performance.",
            "how do i submit my tasks?": "You need to upload your source code to GitHub and submit the repository link via the official submission form shared in your WhatsApp group."
        }
        
        self.questions = list(self.faq_dataset.keys())
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = self.vectorizer.fit_transform(self.questions)
        
        self.assemble_ui()

    def assemble_ui(self):
        brand_frame = tk.Frame(self.root, bg="#161b22", height=70, bd=0, highlightbackground="#30363d", highlightthickness=1)
        brand_frame.pack(fill="x", side="top")
        
        brand_label = tk.Label(brand_frame, text="NEXUS FAQ ENGINE", font=("Consolas", 14, "bold"), fg="#58a6ff", bg="#161b22")
        brand_label.pack(pady=15, side="left", padx=25)
        
        status_tag = tk.Label(brand_frame, text="ONLINE", font=("Arial", 8, "bold"), fg="#238636", bg="#21262d", padx=8, pady=3)
        status_tag.pack(pady=18, side="right", padx=25)

        self.chat_frame = tk.Frame(self.root, bg="#0d1117")
        self.chat_frame.pack(pady=20, fill="both", expand=True, padx=25)

        self.scrollbar = tk.Scrollbar(self.chat_frame, bg="#161b22", bd=0)
        self.scrollbar.pack(side="right", fill="y")

        self.chat_box = tk.Text(self.chat_frame, wrap="word", yscrollcommand=self.scrollbar.set, font=("Segoe UI", 11), bg="#161b22", fg="#c9d1d9", bd=0, highlightbackground="#30363d", highlightthickness=1, padx=15, pady=15)
        self.chat_box.pack(side="left", fill="both", expand=True)
        self.scrollbar.config(command=self.chat_box.yview)

        self.chat_box.tag_config("user_style", foreground="#58a6ff", font=("Segoe UI", 11, "bold"))
        self.chat_box.tag_config("bot_style", foreground="#7ee787", font=("Segoe UI", 11))
        
        self.chat_box.config(state="normal")
        self.chat_box.insert(tk.END, "Bot: Hello! I am your AI FAQ Assistant. How can I help you with your CodeAlpha internship today?\n\n", "bot_style")
        self.chat_box.config(state="disabled")

        input_container = tk.Frame(self.root, bg="#161b22", bd=0, highlightbackground="#30363d", highlightthickness=1)
        input_container.pack(fill="x", padx=25, pady=(0, 25))

        self.user_input = tk.Entry(input_container, font=("Segoe UI", 11), bg="#0d1117", fg="#c9d1d9", bd=0, insertbackground="white")
        self.user_input.pack(side="left", fill="x", expand=True, ipady=10, padx=15, pady=10)
        self.user_input.bind("<Return>", lambda event: self.process_message_pipeline())

        self.send_btn = tk.Button(input_container, text="EXECUTE", font=("Consolas", 10, "bold"), bg="#238636", fg="#ffffff", activebackground="#2ea44f", activeforeground="#ffffff", bd=0, width=12, cursor="hand2", command=self.process_message_pipeline)
        self.send_btn.pack(side="right", padx=15, ipady=6)

    def calculate_nlp_response(self, query):
        query_vector = self.vectorizer.transform([query.lower()])
        similarity_scores = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        highest_score_index = similarity_scores.argmax()
        highest_score = similarity_scores[highest_score_index]
        
        if highest_score > 0.35:
            matched_question = self.questions[highest_score_index]
            return self.faq_dataset[matched_question]
        else:
            return "I'm sorry, I couldn't find an exact match for that query in our local pipeline. Please route your request to services@codealpha.tech."

    def process_message_pipeline(self):
        text_payload = self.user_input.get().strip()
        if not text_payload:
            return
        
        self.chat_box.config(state="normal")
        self.chat_box.insert(tk.END, "You: " + text_payload + "\n", "user_style")
        
        response_payload = self.calculate_nlp_response(text_payload)
        self.chat_box.insert(tk.END, "Bot: " + response_payload + "\n\n", "bot_style")
        
        self.chat_box.config(state="disabled")
        self.chat_box.see(tk.END)
        self.user_input.delete(0, tk.END)

if __name__ == "__main__":
    runtime_environment = tk.Tk()
    application_instance = EnterpriseFAQBot(runtime_environment)
    runtime_environment.mainloop()