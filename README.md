# 🦾🤖 Creating a Neural Network from Scratch

![preview](./assets/BestModelRandSearchCost_Accuracy.png)

This porject goal is to make a general neural network with some of the most important configurations to understand the basic of functionality of a neural network. Test it with different data sets and track it's performance with differents hyperparameters

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

## 📈 Results

The best metrics achieved from mnist hand written numbers using random search were
- Accuracy: 98.37%
- Cost: 0.015

The best metrics achieved from mnist fashion using random search were
- Accuracy: 90%
- Cost: 0.052

## 🧠 Lessons Learn

Understanding linear algebra is a crucial skill for implementing the operations between weights, input values, activations, and bias. These operations are essential for performing the correct feedforward, backpropagation, and weight updates in a neural network

The techniques applied to a neural network are vital for reducing overfitting and enhancing the model's performance. These methods imporve the network's ability to generalize, ensuring that it performs well not only on the training data but alse on new, unseen data.

## 🚀 How to run this project

Follow these steps to run the project on your local machine:

### 1️⃣ Clone the repository
```bash
git clone https://github.com/Miguel9Angel/Neural-Network.git
cd Neural-Network
```

### 2️⃣ Requirements
pip install -r requirements.txt

### 3️⃣ Run the notebook
jupyter notebook notebooks/testing_models.ipynb

## 📁 Repository estructure
```
NEURAL-NETWORK/
├── assets/
│   ├── AccuracyByCostFunction.png
│   ├── AccuracyByInitializer.png
│   ├── AccuracyByLambda.png
│   ├── BestMnistFashionModel.png
│   ├── BestModelRandSearchCost_Accuracy.png
│   ├── Comparing_n.png
│   ├── Costby_n.png
│   ├── Mnist_fashion_firt_model.png
│   └── TestingLearningRateSchedule.png
│
├── data/
│   ├── and_test.csv
│   ├── or_test.csv
│   └── xor_test.csv
│
├── notebooks/
│   └── testing_models.ipynb
│
├── src/
│   ├── __pycache__/
│   └── network.py
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