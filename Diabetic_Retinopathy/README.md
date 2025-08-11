
Project Overview: Tackling Diabetic Retinopathy with an Ensemble
For this project, I dove into the critical task of detecting Diabetic Retinopathy (DR), a serious eye condition that can affect people with diabetes. The goal was to classify retinal images into one of five stages of severity: No DR, Mild, Moderate, Severe, or Proliferative DR. To do this, I used the huge EyePACS dataset from Kaggle(https://www.kaggle.com/c/diabetic-retinopathy-detection/), which contains thousands of high-resolution retinal fundus images.

My main idea wasn't just to use one model, but to see how effectively I could combine the strengths of several different pre-trained models. I figured that by creating an ensembled model, the final prediction would be more robust.

I picked a team of four well-known models: two "heavy-weighted" (ResNet-50 and DenseNet-121) and two "light-and-fast" models (EfficientNet and MobileNet). To adapt them to this specific task, I froze the early layers (which detect basic features like edges and colors) and only fine-tuned the last 20-30% of each model on the DR dataset.

Once each model was performing reasonably well on its own, I brought them all together. I built a four-branch neural network where the penultimate features from each model were concatenated and then fed into a small "meta-model" for the final classification. This ensemble approach worked, giving me a nice, though modest, 2% increase in accuracy over the individual models.

Navigating Real-World Challenges:
Right from the start, I hit a classic data science roadblock: a massive dataset (30-40k images!) and limited computing power (just the two Kaggle T4 GPUs). This meant I couldn't just throw the entire dataset at the problem. As you can see in my preprocessing notebook, I experimented with various subsets of the data (5k, 10k, and 12k images) and played with balancing the classes.

Ultimately, I settled on a 12k image subset that was balanced using SMOTE. This gave me the best trade-off between manageable training times and a decent model performance.

An Interesting Discovery and a Pragmatic Solution:
After training the models, I took a hard look at the confusion matrix. I noticed a clear pattern: my model was getting really confused between the "No DR," "Mild," and "Moderate" classes, causing them to overlap quite a bit. It was, however, much better at identifying the more severe cases.

Given the time constraints, I opted for a pragmatic solution. I decided to merge the "No DR" and "Mild DR" categories into a single class. It was a bit of a brute-force fix, but it worked! After retraining the models on this simplified task, I saw a significant 5% improvement in overall performance.

All the steps I took to clean, resize, and prepare the data for this project are detailed in the preprocess.ipynb notebook.