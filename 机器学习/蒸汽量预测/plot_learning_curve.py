import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import numpy as np

def plot_learning_error(algo, X_train, X_test, y_train, y_test):
        """绘制学习曲线：只需要传入算法(或实例对象)、X_train、X_test、y_train、y_test"""
        """当使用该函数时传入算法，该算法的变量要进行实例化，如：PolynomialRegression(degree=2)，变量 degree 要进行实例化"""
        train_score = []
        test_score = []
        for i in range(10, len(X_train)+1, 10):
                algo.fit(X_train[:i], y_train[:i])
        
                y_train_predict = algo.predict(X_train[:i])
                train_score.append(mean_squared_error(y_train[:i], y_train_predict))

                y_test_predict = algo.predict(X_test)
                test_score.append(mean_squared_error(y_test, y_test_predict))

        plt.plot([i for i in range(1, len(train_score)+1)],
                train_score, label="train")
        plt.plot([i for i in range(1, len(test_score)+1)],
                test_score, label="test")

        plt.legend()
        plt.show()

def plot_learning_score(estimator, title, X, y, ylim=None, cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0),n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
        plt.figure()
        plt.title(title)
        if ylim is not None:
                plt.ylim(*ylim)
        plt.xlabel("Training examples")
        plt.ylabel("Score")
        train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        print(train_scores_mean)
        print(test_scores_mean)

        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1,
                        color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
                label="Training score")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
                label="Cross-validation score")

        plt.legend(loc="best")
        return plt