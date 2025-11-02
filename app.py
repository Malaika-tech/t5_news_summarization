# ==============================================
# Streamlit App for T5 News Summarization
# ==============================================
import streamlit as st
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
import torch

# ==============================================
# CONFIGURATION
# ==============================================
MODEL_NAME = "MalaikaNaveed1/t5_news_summarization"  # Your Hugging Face model repo
MAX_INPUT_LEN = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==============================================
# LOAD MODEL AND TOKENIZER
# ==============================================
@st.cache_resource
def load_model():
    st.write("🔄 Loading fine-tuned T5 model from Hugging Face Hub...")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.to(DEVICE)
    return model, tokenizer

model, tokenizer = load_model()
st.success("✅ Model loaded successfully from Hugging Face!")

# ==============================================
# APP TITLE & DESCRIPTION
# ==============================================
st.title("📰 T5 News Summarization App")
st.markdown("""
This app uses a fine-tuned **T5 model** to summarize long news articles from the **CNN/DailyMail** dataset.

**Instructions:**
1. Paste a news article into the input box below.
2. Adjust summary length or decoding parameters (optional).
3. Click **Summarize** to generate the abstractive summary.
""")

# ==============================================
# SIDEBAR SETTINGS
# ==============================================
st.sidebar.header("⚙️ Generation Settings")
max_length = st.sidebar.slider("Maximum Summary Length", 50, 300, 128)
num_beams = st.sidebar.slider("Beam Search Width", 2, 10, 4)
length_penalty = st.sidebar.slider("Length Penalty", 0.5, 3.0, 2.0, 0.1)
do_sample = st.sidebar.checkbox("Enable Sampling (for creative summaries)", value=False)
temperature = st.sidebar.slider("Sampling Temperature", 0.7, 1.5, 1.0, 0.1)

# ==============================================
# INPUT TEXT AREA
# ==============================================
article_text = st.text_area(
    "📝 Paste your news article below:",
    height=250,
    placeholder="Enter a news article to summarize..."
)

# ==============================================
# SUMMARIZATION FUNCTION
# ==============================================
def generate_summary(article):
    """Generate summary using the fine-tuned T5 model."""
    input_text = "summarize: " + article.strip()
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        max_length=MAX_INPUT_LEN,
        truncation=True
    ).to(DEVICE)

    with torch.no_grad():
        summary_ids = model.generate(
            inputs["input_ids"],
            max_length=max_length,
            num_beams=num_beams,
            length_penalty=length_penalty,
            do_sample=do_sample,
            temperature=temperature,
            early_stopping=True
        )

    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ==============================================
# ROUGE SCORE CALCULATION
# ==============================================
def compute_rouge(reference, generated):
    """Compute ROUGE scores for evaluation."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, generated)
    return {
        "ROUGE-1": round(scores["rouge1"].fmeasure, 4),
        "ROUGE-2": round(scores["rouge2"].fmeasure, 4),
        "ROUGE-L": round(scores["rougeL"].fmeasure, 4)
    }

# ==============================================
# BUTTON ACTION
# ==============================================
if st.button("🚀 Summarize"):
    if not article_text.strip():
        st.warning("⚠️ Please enter an article first.")
    else:
        with st.spinner("Generating summary... please wait ⏳"):
            summary = generate_summary(article_text)

        st.subheader("📋 Generated Summary:")
        st.success(summary)

        # Optional: reference text for comparison
        reference_text = st.text_area(
            "🧾 (Optional) Reference Summary (for evaluation):",
            placeholder="Paste the ground-truth summary here to compute ROUGE metrics..."
        )

        if reference_text.strip():
            rouge = compute_rouge(reference_text, summary)
            st.subheader("📊 ROUGE Evaluation:")
            st.json(rouge)

# ==============================================
# FOOTER
# ==============================================
st.markdown("---")
st.markdown("""
**Developed for Task 2 — Encoder-Decoder Model (T5) Fine-tuning Project**  
📚 Dataset: CNN/DailyMail  
🧠 Model: [T5-small / T5-base](https://huggingface.co/t5-base)  
🤗 Hosted on [Hugging Face Hub](https://huggingface.co/MalaikaNaveed1/t5_news_summarization)  
🧑‍💻 Built with Streamlit & Hugging Face Transformers
""")
