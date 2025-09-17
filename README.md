# 🦾🤖 Creating a Neural Network from Scratch

![preview](./assets/BestModelRandSearchCost_Accuracy.png)

Make a general convolutional neural network from scratch with some of the most important configurations. Implementing the convolutional layer, pooling layer, flatten layer and the dense layer to undertand the altgorithm. Then test it with different data sets and track it's performance with differents hyperparameters.

## 📊 Dataset

- Source: [mnist handwritten numbers] (http://yann.lecun.com)
- Nº of records: 70.000 image numbers
- Variables: pixels image, labels

## 🛠️ Techniques Used

- Análisis exploratorio de datos (EDA)
- Neural Network
- Numpy matrix operations
- Backpropagation
- Regularization
- Categorical cross-entropy
- Convolutional Neural Network

## 📈 Results

The best metrics achieved from mnist hand written numbers using random search were
- Accuracy: 99%
- Cost: 0.009


## 🧠 Lessons Learn

Understanding linear algebra is a crucial skill for implementing the matricial operations between kernels, input values, activations, and bias nad other techniques like maxpooling. These operations are essential for performing the correct feedforward, backpropagation, and weight updates in a convolutional neural network.

The techniques applied to a convolutional neural network are vital for reducing overfitting and enhancing the model's performance. These methods improve the network's ability to generalize, ensuring that it performs well not only on the training data but also on new, unseen data.

## 🚀 How to run this project

Follow these steps to run the project on your local machine:

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Miguel9Angel/Convolutional-Neural-Network.git
cd Convolutional-Neural-Network
```

### 2️⃣ Requirements
pip install -r requirements.txt

### 3️⃣ Run the notebook
jupyter notebook notebooks/testing_models.ipynb

## 📁 Repository estructure
```
CONVOLUTIONAL-NEURAL-NETWORK/

├── notebooks/
│   └── testing_models.ipynb
│
├── src/
│   ├── __pycache__/
│   └── convolutional_net.py
│
├── LICENSE
├── README.md
└── requirements.txt
```
## 📜 License

This project is licensed under the [Licencia MIT](./LICENSE).  
You are free to use, modify, and distribute this code, provided that proper credit is given.

--------------------------------------------------------------------------------------

## 🙋 About me

My name is Miguel Angel Soler Otalora, a mechanical engineer with a background in data science and artificial intelligence. I combine the analytical and structured thinking of engineering with modern skills in data analysis, visualization, and predictive modeling.

This project is part of my portfolio to apply for roles as a Data Analyst or Data Scientist, and it reflects my interest in applying data analysis to real-world problems.

📫 You can contact me on [LinkedIn](https://linkedin.com/in/miguel-soler-ml) or explore more projects on [GitHub](https://github.com/Miguel9Angel).