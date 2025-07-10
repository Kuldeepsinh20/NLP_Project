
# The full code is pasted here from the user's input without modification.
# In a real-world scenario, minor adjustments may be needed based on local environment setup.
# This code uses tkinter, transformers, and PIL to build a desktop-based NLP app.

# [Insert user's full code from above here, which is already complete and well-structured.]

import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from PIL import Image, ImageTk
import threading
import io
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForTokenClassification, AutoModelForMaskedLM, AutoProcessor
import torch

class ModernNLPApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Multi-Task NLP Professional Suite")
        self.root.geometry("1000x700")
        self.root.configure(bg='#f0f0f0')

        self.setup_styles()
        self.init_models()
        self.create_ui()

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        bg_color = '#ffffff'
        accent_color = '#2196F3'
        success_color = '#4CAF50'
        text_color = '#333333'
        style.configure('Title.TLabel', font=('Segoe UI', 16, 'bold'), foreground=text_color, background=bg_color)
        style.configure('Subtitle.TLabel', font=('Segoe UI', 10), foreground='#666666', background=bg_color)
        style.configure('Modern.TButton', font=('Segoe UI', 10, 'bold'), foreground='white', background=accent_color, borderwidth=0, focuscolor='none', relief='flat')
        style.map('Modern.TButton', background=[('active', '#1976D2'), ('pressed', '#0D47A1')])
        style.configure('Success.TButton', font=('Segoe UI', 10, 'bold'), foreground='white', background=success_color, borderwidth=0, focuscolor='none', relief='flat')
        style.map('Success.TButton', background=[('active', '#45a049'), ('pressed', '#3d8b40')])
        style.configure('Modern.TNotebook', background=bg_color, borderwidth=0)
        style.configure('Modern.TNotebook.Tab', font=('Segoe UI', 10, 'bold'), padding=[20, 10], background='#e0e0e0', foreground=text_color)
        style.map('Modern.TNotebook.Tab', background=[('selected', accent_color), ('active', '#e3f2fd')], foreground=[('selected', 'white')])

    def init_models(self):
        try:
            self.show_loading_message("Loading NLP models...")
            self.env_classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
            self.ner = pipeline("ner", grouped_entities=True)
            self.mask_filler = pipeline("fill-mask", model="bert-base-uncased")
            try:
                self.image_gen = None  # Optional placeholder
            except:
                self.image_gen = None
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load models: {str(e)}")

    def show_loading_message(self, message):
        print(f"Loading: {message}")

    def create_ui(self):
        main_frame = ttk.Frame(self.root, style='Modern.TFrame')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        title_label = ttk.Label(main_frame, text="🌿 Multi-Task NLP Professional Suite", style='Title.TLabel')
        title_label.pack(pady=(0, 10))
        subtitle_label = ttk.Label(main_frame, text="Advanced Natural Language Processing & AI Tools", style='Subtitle.TLabel')
        subtitle_label.pack(pady=(0, 20))
        self.notebook = ttk.Notebook(main_frame, style='Modern.TNotebook')
        self.notebook.pack(fill=tk.BOTH, expand=True)
        self.create_env_classifier_tab()
        self.create_image_gen_tab()
        self.create_ner_tab()
        self.create_mask_fill_tab()

    def create_env_classifier_tab(self):
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="🌍 Environment Classifier")
        content_frame = ttk.Frame(tab_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        title = ttk.Label(content_frame, text="Environmental Text Classification", style='Title.TLabel')
        title.pack(pady=(0, 10))
        input_frame = ttk.LabelFrame(content_frame, text="Input Text", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        self.env_text_input = scrolledtext.ScrolledText(input_frame, height=4, font=('Segoe UI', 10), wrap=tk.WORD)
        self.env_text_input.pack(fill=tk.X, pady=(0, 10))
        analyze_btn = ttk.Button(input_frame, text="🔍 Analyze Text", style='Modern.TButton', command=self.classify_environment)
        analyze_btn.pack()
        results_frame = ttk.LabelFrame(content_frame, text="Classification Results", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.env_results = scrolledtext.ScrolledText(results_frame, height=6, font=('Segoe UI', 10), wrap=tk.WORD, state=tk.DISABLED)
        self.env_results.pack(fill=tk.BOTH, expand=True)

    def create_image_gen_tab(self):
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="🎨 Text to Image")
        content_frame = ttk.Frame(tab_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        title = ttk.Label(content_frame, text="Text-to-Image Generation", style='Title.TLabel')
        title.pack(pady=(0, 10))
        input_frame = ttk.LabelFrame(content_frame, text="Image Prompt", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        self.image_prompt_input = scrolledtext.ScrolledText(input_frame, height=3, font=('Segoe UI', 10), wrap=tk.WORD)
        self.image_prompt_input.pack(fill=tk.X, pady=(0, 10))
        generate_btn = ttk.Button(input_frame, text="🎨 Generate Image", style='Success.TButton', command=self.generate_image)
        generate_btn.pack()
        image_frame = ttk.LabelFrame(content_frame, text="Generated Image", padding=10)
        image_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.image_label = ttk.Label(image_frame, text="Generated image will appear here", font=('Segoe UI', 10), foreground='#666666')
        self.image_label.pack(expand=True)

    def create_ner_tab(self):
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="🏷️ Named Entities")
        content_frame = ttk.Frame(tab_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        title = ttk.Label(content_frame, text="Named Entity Recognition", style='Title.TLabel')
        title.pack(pady=(0, 10))
        input_frame = ttk.LabelFrame(content_frame, text="Input Text", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        self.ner_text_input = scrolledtext.ScrolledText(input_frame, height=4, font=('Segoe UI', 10), wrap=tk.WORD)
        self.ner_text_input.pack(fill=tk.X, pady=(0, 10))
        extract_btn = ttk.Button(input_frame, text="🏷️ Extract Entities", style='Modern.TButton', command=self.extract_entities)
        extract_btn.pack()
        results_frame = ttk.LabelFrame(content_frame, text="Extracted Entities", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.ner_results = scrolledtext.ScrolledText(results_frame, height=6, font=('Segoe UI', 10), wrap=tk.WORD, state=tk.DISABLED)
        self.ner_results.pack(fill=tk.BOTH, expand=True)

    def create_mask_fill_tab(self):
        tab_frame = ttk.Frame(self.notebook)
        self.notebook.add(tab_frame, text="🎭 Fill Mask")
        content_frame = ttk.Frame(tab_frame)
        content_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        title = ttk.Label(content_frame, text="Masked Language Model", style='Title.TLabel')
        title.pack(pady=(0, 10))
        instructions = ttk.Label(content_frame, text="Enter text with [MASK] tokens (e.g., 'The earth is [MASK].')", style='Subtitle.TLabel')
        instructions.pack(pady=(0, 10))
        input_frame = ttk.LabelFrame(content_frame, text="Masked Text", padding=10)
        input_frame.pack(fill=tk.X, pady=(0, 20))
        self.mask_text_input = scrolledtext.ScrolledText(input_frame, height=3, font=('Segoe UI', 10), wrap=tk.WORD)
        self.mask_text_input.pack(fill=tk.X, pady=(0, 10))
        predict_btn = ttk.Button(input_frame, text="🎭 Predict Mask", style='Success.TButton', command=self.fill_mask)
        predict_btn.pack()
        results_frame = ttk.LabelFrame(content_frame, text="Predictions", padding=10)
        results_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.mask_results = scrolledtext.ScrolledText(results_frame, height=6, font=('Segoe UI', 10), wrap=tk.WORD, state=tk.DISABLED)
        self.mask_results.pack(fill=tk.BOTH, expand=True)

    def update_results(self, text_widget, content):
        text_widget.config(state=tk.NORMAL)
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, content)
        text_widget.config(state=tk.DISABLED)

    def classify_environment(self):
        text = self.env_text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return
        try:
            keywords = ["climate", "pollution", "earth", "global warming", "deforestation", "recycle", "environment", "sustainability", "carbon", "emissions", "renewable", "biodiversity"]
            score = sum(1 for kw in keywords if kw in text.lower())
            confidence = min(score / 3, 1.0)
            if score > 0:
                result = f"Classification: Environment-related\nConfidence: {confidence:.2f}\nKeywords found: {score}\n\n"
                result += f"Environmental keywords detected: {', '.join([kw for kw in keywords if kw in text.lower()])}"
            else:
                result = f"Classification: Not Environment-related\nConfidence: {1-confidence:.2f}\n\nNo environmental keywords detected."
            self.update_results(self.env_results, result)
        except Exception as e:
            messagebox.showerror("Error", f"Classification failed: {str(e)}")

    def generate_image(self):
        prompt = self.image_prompt_input.get(1.0, tk.END).strip()
        if not prompt:
            messagebox.showwarning("Warning", "Please enter an image prompt.")
            return
        self.image_label.config(text=f"Generating image for: '{prompt}'\n\n(Image generation placeholder)")

    def extract_entities(self):
        text = self.ner_text_input.get(1.0, tk.END).strip()
        if not text:
            messagebox.showwarning("Warning", "Please enter some text to analyze.")
            return
        try:
            entities = self.ner(text)
            if entities:
                result = "Extracted Entities:\n" + "="*50 + "\n\n"
                for i, entity in enumerate(entities, 1):
                    result += f"{i}. {entity['entity_group']}: {entity['word']}\n"
                    result += f"   Confidence: {entity['score']:.3f}\n\n"
            else:
                result = "No named entities found in the text."
            self.update_results(self.ner_results, result)
        except Exception as e:
            messagebox.showerror("Error", f"Entity extraction failed: {str(e)}")

    def fill_mask(self):
        text = self.mask_text_input.get(1.0, tk.END).strip()
        if not text or "[MASK]" not in text:
            messagebox.showwarning("Warning", "Please include [MASK] in your text.")
            return
        try:
            predictions = self.mask_filler(text)
            result = "Mask Predictions:\n" + "="*50 + "\n\n"
            for i, pred in enumerate(predictions, 1):
                result += f"{i}. {pred['sequence']}\n"
                result += f"   Confidence: {pred['score']:.3f}\n"
                result += f"   Token: {pred['token_str']}\n\n"
            self.update_results(self.mask_results, result)
        except Exception as e:
            messagebox.showerror("Error", f"Mask filling failed: {str(e)}")

def main():
    root = tk.Tk()
    app = ModernNLPApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
