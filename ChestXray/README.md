This project sits at the exciting crossroads of computer vision and NLP, with an ambitious goal: teaching an AI model to generate a radiology report just by looking at a chest X-ray. For this adventure, I used the Padchest dataset from Kaggle(https://www.kaggle.com/datasets/seoyunje/padchest-small-dataset), a massive collection of 160,000 X-rays. The first interesting challenge? All the corresponding reports were in Spanish.

First Hurdle: Breaking the Language Barrier
Before I could even think about image-to-text models, I had to get the reports into English. My first thought was to use a standard machine translation model like Facebook's NLLB ("No Language Left Behind"). I was curious to see how a general-purpose tool would handle highly specific medical language it wasn't trained on. As I suspected, the results were a bit clunky. The translations struggled with complex medical terminologies and often sounded unnatural.

This led me to my next experiment: what if I used a smarter, instruction-tuned LLM like Mistral Instruct to not just translate, but to refine the text? (I got this idea from Automatic Speech Recognition, sometimes smaller or older ASR model's outputs are refined by passing them through an LLM) So I gave it a very specific role to play with a detailed prompt:

"You are a board-certified radiologist... your task is to revise the following chest X-ray report... correct all grammatical errors, use standard medical terminology, and crucially, DO NOT invent any new medical findings..."

The difference was night and day! I double-checked a few outputs by having ChatGPT do a direct translation of the original Spanish, and Mistral's refined versions were significantly clearer and more professional.

The Big Experiment: Centralized vs. Federated Learning
With my English text-image pairs ready, it was time to build the core model. Given my limited compute power (Kaggle T4s), I couldn't use the entire set of 160k images. I filtered the dataset down to a more manageable ~9,400 images and chose a lightweight architecture: a ViT-Tiny to "see" the image and a DistilGPT2 to "write" the report, ofcourse with a projection module in between consisting of a single linear layer, mapping both the encoder and decoder.

Here, I wanted to compare two key approaches:

Centralized Training: The standard method where I trained one model on the entire dataset at once.

Federated Learning: A simulation highly relevant for medical data. I split the dataset between two clients, trained a model for each locally, and then averaged their weights using FedAvg. This mimics a real-world scenario where hospitals could collaborate without ever sharing private patient data.

Honest Results and the Key Takeaway:
Now, I'll be honest—the final generated reports weren't ready to be sent to a doctor. The quality was a bit rough, which I chalk up to two main things: the dataset being heavily skewed towards normal cases ("No significant findings") and the relatively small slice of data I was able to use.

But here’s the most important insight from the project: both the centralized and federated learning approaches produced very similar results and learning curves.

This is a huge deal because it strongly suggests that we can build powerful, collaborative medical AI systems using federated learning. Hospitals could improve their models by sharing their learnings, not their sensitive patient data. It's a promising step forward for building effective and privacy-preserving AI in healthcare.

For anyone who wants to dive into the technical details, the data preparation steps are in the PADCHEST_FILTERING.ipynb and kaggle_npy.ipynb notebooks, translation workflow in Es2En.ipynb and Report Generation in main.ipynb notebooks!
