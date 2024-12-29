//  const { SimpleLinearRegression } = require("ml-regression-simple-linear");
// import { SimpleLinearRegression } from "ml-regression-simple-linear";

// let inputs = [80, 60, 10, 20, 30];
// let outputs = [20, 40, 30, 50, 60];

// console.log(regression.predict(80));
// console.log(regression.coefficients);
// console.log(regression.toString(3));

// Predecting Treatment Using Logistic regression
import { Matrix } from "ml-matrix";

export default class LogisticRegressionTwoClasses {
  constructor(options = {}) {
    this.numSteps = options.numSteps || 500000;
    this.learningRate = options.learningRate || 5e-4;
    this.weights = options.weights ? Matrix.checkMatrix(options.weights) : null;
  }

  train(features, target) {
    var weights = Matrix.zeros(1, features.columns);

    for (var step = 0; step < this.numSteps; step++) {
      var scores = features.mmul(weights.transpose());
      var predictions = sigmoid(scores);

      //update weight with gradients
      var outputErrorSignal = Matrix.columnVector(predictions);
      var gradients = features.transpose().mmul(outputErrorSignal);
      weights = weights.add(gradients.mul(this.learningRate).transpose());
    }
    this.weights = weights;
  }

  testScores(features) {
    var finalData = features.mmul(this.weights.transpose());
    var predictions = sigmoid(finalData);
    predictions = Matrix.columnVector(predictions);
    return predictions.to1DArray();
  }

  predict(features) {
    var finalData = features.mmul(this.weights.transpose());
    var predictions = sigmoid(finalData);
    predictions = Matrix.columnVector(predictions);
    predictions = predictions.map((value) => Math.round(value));
    return predictions.to1DArray();
  }

  static load(model) {
    return new LogisticRegressionTwoClasses(model);
  }

  toJSON() {
    return {
      numSteps: this.numSteps,
      learningRate: this.learningRate,
      weights: this.weights,
    };
  }
}

function sigmoid(scores) {
  return Matrix.map(scores, (value) => 1 / (1 + Math.exp(-value)));
}

// our training set (X,Y)
var X = new Matrix([
  [1, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
]);
var Y = Matrix.columnVector([1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1]);

// the test set (Xtest , Ytest)
var Xtest = new Matrix([
  [1, 0, 0],
  [0, 0, 0],
  [0, 0, 0],
  [0, 1, 0],
  [0, 0, 1],
]);
var Ytest = Matrix.columnVector([
  1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
]);

// we will train our model
let logreg = new LogisticRegressionTwoClasses();
logreg.train(X, Y);

//we will try to predict the test set
var finalResults = logreg.predict(Xtest);

// now you can compare final results with the Ytest
var count = 0;
for (var i = 0; i < finalResults.length; i++) {
  if (finalResults[i] === Ytest.get(i, 0)) {
    count = count + 1;
  }
}
// let regression = new SimpleLinearRegression(inputs, outputs);

console.log(regression.predict(80));

// Predecting Salaries after College Using Linear Regression
document.getElementById("hello").innerHTML = "Using ml-regression";
const Xtrain = [78, 70.06, 70, 74.64, 73.9, 76.32, 72.98, 8.58];
const Ytrain = [7855, 7359, 7705, 7614, 8347, 8619, 7356, 1587];

//building a model
let regression = new SimpleLinearRegression(Xtrain, Ytrain);

// console.log((count / finalResults.length) * 100);
