# Advertisement Dataset Analysis

## Overview

This project involves the analysis of an advertisement dataset to predict user behavior. We performed Exploratory Data Analysis (EDA), followed by building and optimizing classification models for the 'Click' column and regression models for the 'Age' column.
### Update: In addition to traditional machine learning models, we also explored deep learning techniques to further enhance our predictive capabilities.

## Dataset

The advertisement dataset includes various features such as:

- `Time_spent`: Time spent by the user on the advertisement
- `Age`: Age of the user
- `Income`: Income of the user
- `Net_Usage`: Internet usage of the user
- `Ad Topic Line`: Topic of the advert
- `City`: City of the user
- `Male`: Gender of the user (1 for male, 0 for female)
- `Country`: Country of the user
- `Timestamp`: Timestamp of the user
- `Clicked`: Whether the user clicked on the advertisement (1 for yes, 0 for no)

## Analysis and Modeling

### Exploratory Data Analysis (EDA)

We conducted a comprehensive EDA to understand the dataset, identify patterns, and preprocess the data for modeling.

### Classification on 'Click' Column

We built and optimized classification models to predict whether a user clicked on an advertisement. The best model was a Random Forest classifier with GridSearch optimization, achieving an accuracy of over 0.96.

### Regression on 'Age' Column

Despite the dataset not being ideally suited for linear modeling, we challenged ourselves with regression tasks to predict the age of users. The best results were achieved using a Random Forest Regressor optimized with GridSearch, yielding an R² of 0.38 and an RMSE of 6.57.

## Results

- **Classification Model**: 
  - Best Model: Random Forest classifier with GridSearch
  - Accuracy: > 0.96

- **Regression Model**: 
  - Best Model: Random Forest Regressor with GridSearch
  - R²: 0.38
  - RMSE: 6.57

## Conclusion

This project showcases the application of machine learning techniques to predict user behavior in an advertisement context. The high accuracy of the classification model demonstrates the effectiveness of Random Forest classifiers for this task. The regression task, though challenging, provided valuable insights and highlighted the potential limitations of the dataset.

## Requirements

- Python 3.7+
- Jupyter Notebook
- Pandas
- NumPy
- Scikit-learn
- Seaborn
- Matplotlib
- Tensorflow

## Deep Learning Update
In addition to traditional machine learning models, we incorporated deep learning techniques to further enhance our predictive capabilities for both classification and regression tasks.

## Deep Learning for Classification

We implemented a deep learning model using TensorFlow to predict whether a user clicked on an advertisement. The neural network architecture included multiple layers with the ReLU activation function, and the Adam optimizer was used to adjust weights efficiently during training. We monitored the model's performance using model.history.history to plot learning curves and check for overfitting. This approach allowed us to capture more complex patterns in the data, potentially boosting classification performance.

## Deep Learning for Regression

For the regression task, we used deep learning to predict the age of users. A neural network was built using TensorFlow, employing ReLU activation functions and the Adam optimizer. We carefully monitored the training process using model.history.history to detect and address any overfitting. This deep learning model aimed to uncover intricate relationships within the data that traditional regression models might overlook.

## Results and Impact
 - Classification with Deep Learning: The deep learning model provided a robust alternative to the Random Forest classifier, effectively capturing complex interactions between features.
 - Regression with Deep Learning: Although challenging, the deep learning regression model offered a nuanced approach to predict age, requiring careful tuning to avoid overfitting and produce meaningful results.

This deep learning integration showcased the potential for neural networks to enhance prediction accuracy, particularly in complex datasets like this advertisement dataset.

## Acknowledgments

Special thanks to the creators of the dataset and the developers of the Python libraries used in this project.

## Contributing

Contributions are welcome! Please fork this repository and submit a pull request for any improvements or bug fixes.

---

