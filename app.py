import gradio as gr
from predict import predict_sentiment

# Create the Gradio interface
iface = gr.Interface(
    fn=predict_sentiment,             # Directly use predict_sentiment
    inputs=gr.Textbox(
        lines=10, 
        placeholder="Paste the transcription of the video here...", 
        label="Video Transcription",
        interactive=True,  # Make it interactive for user input
    ),
    outputs=gr.Label(label="Sentiment Prediction"),  # Use a label for output
    title="Video Sentiment Analysis",  # Title of the interface
    description="Enter the transcription of a video to predict its sentiment.",
    theme="compact",  # Use a compact theme for a cleaner look
    css="""
    .gradio-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        min-height: 100vh;  /* Ensures the content is vertically centered */
    }
    .input-textbox, .output-label {
        margin: 10px;  /* Adds spacing between the input and output boxes */
        width: 80%;  /* Makes the boxes responsive */
        max-width: 800px;  /* Limits the maximum width for large screens */
    }
    .footer { display: none; }  /* Hides the footer */
    .input-textbox { 
        border: 2px solid #4A90E2; 
        border-radius: 8px; 
    }  /* Style the input box */
    .output-label { 
        font-size: 18px; 
        font-weight: bold; 
        color: #333; 
    }  /* Style the output label */
    """,  # Add custom CSS for additional styling
)

# Launch the Gradio interface
if __name__ == "__main__":
    iface.launch(share=True)
