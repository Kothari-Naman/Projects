In this  project, i considered BRATS 2020 Challenge dataset from Kaggle(https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation) and tried to perform a binary Tumor segmentation tasks. I compared two approaches and observed similar results, one with a normal centralized training and other in a Federated Learning Simulation - Splitting the data into 3 Clients and then training a local model on that subset for each client and finally taking average of the weights of all 3 clients using FedAvg function.

For this project, I  specifically focused on brain tumor segmentation. I used the BRATS 2020 Challenge dataset, which I grabbed from Kaggle, to see if I could train a model to accurately identify and Segment tumor regions in brain scans (Binary masks).

My main goal was to compare two different ways of training a small 3D U-Net Style model for this binary segmentation task.

First, I went with the classic, centralized approach. This is the standard method where you train a single model on the entire dataset all at once. It's straightforward and gave me a solid performance baseline.

Then, I explored a more modern and privacy-focused technique: Federated Learning. I simulated a real-world scenario by splitting the dataset among three separate "clients." Each client trained its own local model on just its slice of the data. After the local training was done, I used the Federated Averaging (FedAvg) algorithm to combine what each model had learned by simply averaging their weights.

Interestingly, both methods ended up giving me very similar IOU scores. The federated learning setup mimics training on decentralized data without ever sharing it, so rather than the data going to the model, the model ends up going to the Data.   

The preprocessing notebook describes the steps i followed to utilize BRATS data from Kaggle.  