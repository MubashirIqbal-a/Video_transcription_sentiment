import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig

def load_model_and_tokenizer(model_dir):
    """
    Load the pre-trained model and tokenizer from the specified directory.
    """
    config = AutoConfig.from_pretrained(model_dir)
    config.num_labels = 2
    
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, config=config)
    return tokenizer, model

def predict_sentiment(text):
    """
    Predict the sentiment of the input text using the loaded model and tokenizer.
    """
    model_dir = "models/bart"
    tokenizer, model = load_model_and_tokenizer(model_dir)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()
    return "Positive" if predicted_class_id == 1 else "Negative"

# negative_samples = [
#     "This product has been a huge disappointment. From the moment I unboxed it, I could tell it wasn't going to live up to the hype. The build quality is poor, and it feels cheap. The performance is slow and glitchy, making it frustrating to use. Customer support was unhelpful and dismissive when I reached out for assistance. I regret spending my money on this, and I strongly advise others to avoid it. Overall, this has been a very negative experience, and I would not recommend this product to anyone.",
    
#     "I had high hopes for this service, but it turned out to be a waste of time and money. The interface is confusing, and the features are severely lacking. Every time I tried to get help, I was met with automated responses that didn't solve my problem. I'm extremely dissatisfied and will be canceling my subscription immediately. I do not recommend this service at all.",
    
#     "This movie was a total letdown. The plot was incoherent, the characters were poorly developed, and the acting was subpar. I expected a thrilling experience, but instead, I was bored and annoyed. It was a waste of two hours that I will never get back. Save yourself the trouble and skip this one.",
    
#     "The food at this restaurant was terrible. The dishes were bland, the portions were small, and the service was incredibly slow. I had to wait over an hour for my meal, only to be served cold, tasteless food. I left feeling hungry and disappointed. I would not eat here again.",
    
#     "This app is useless. It crashes constantly, and when it does work, it is incredibly slow. The features that are advertised are either missing or don't work properly. I've contacted support multiple times, but they have not been helpful. I'm deleting this app and looking for a better alternative. Do not download this app."
# ]
# for i in negative_samples:
#     print(predict_sentiment(i))