# 🐶🐱 Dog vs Cat Classifier with CAM (Class Activation Map)

This project is a convolutional neural network (CNN)-based **binary image classifier** to distinguish between images of **dogs** and **cats**, built using **TensorFlow/Keras**.

What makes this project special is the use of **Class Activation Maps (CAMs)** to visualize the **discriminative regions** of an image that the model focuses on while making predictions. This enhances the interpretability of deep learning models.

---

## 📂 Dataset

We use the **Dogs vs Cats** dataset from Kaggle.

### ✅ To download the dataset via code, follow these steps:

1. Go to your Kaggle account settings and download your **API token**:
   - Visit: https://www.kaggle.com/account
   - Click on “Create New API Token” — this will download a file called `kaggle.json`

2. Place the `kaggle.json` file inside a folder named `~/.kaggle/` (or configure it for your environment as shown below).

### 📥 Setup inside a Colab/Jupyter notebook:

```python
import os
import zipfile

# Upload kaggle.json
from google.colab import files
files.upload()  # Choose the kaggle.json file

# Create Kaggle folder and move file
os.makedirs("/root/.kaggle", exist_ok=True)
!mv kaggle.json /root/.kaggle/kaggle.json
!chmod 600 /root/.kaggle/kaggle.json

# Download and unzip dataset
!kaggle competitions download -c dogs-vs-cats
with zipfile.ZipFile("dogs-vs-cats.zip", 'r') as zip_ref:
    zip_ref.extractall("dataset")
