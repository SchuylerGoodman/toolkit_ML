
#ifndef PERCEPTRON_H
#define PERCEPTRON_H


#include "matrix.h"
#include "rand.h"
#include "learner.h"
#include "error.h"
#include "time.h"
#include <iostream>
#include <vector>

class Perceptron : public SupervisedLearner
{
private:
	Rand m_rand; // pseudo-random number generator
    int maxEpochs;
    double learningRate;
    bool thresholdPrediction;

    std::vector<double> weights;
    double biasAttr;

    static const int MAX_EPOCHS = 50;
    static const double LEARNING_RATE = 0.5;

public:
    Perceptron()
    : SupervisedLearner(), maxEpochs(MAX_EPOCHS), learningRate(LEARNING_RATE), thresholdPrediction(true)
    {
    }

    Perceptron(Rand r, int maxEpochs = MAX_EPOCHS, double learningRate = LEARNING_RATE, bool thresholdPrediction = true)
    : SupervisedLearner(), m_rand(r), maxEpochs(maxEpochs), learningRate(learningRate), thresholdPrediction(thresholdPrediction)
    {
    }

	~Perceptron()
	{
	}

    Perceptron(const Perceptron& p)
    :   m_rand(p.m_rand), maxEpochs(p.maxEpochs), learningRate(p.learningRate),
        thresholdPrediction(p.thresholdPrediction), weights(p.weights), biasAttr(p.biasAttr)
    {
    }
    
    Perceptron& operator=(const Perceptron& rhs)
    {
        m_rand = rhs.m_rand;
        maxEpochs = rhs.maxEpochs;
        learningRate = rhs.learningRate;
        thresholdPrediction = rhs.thresholdPrediction;
        weights = std::vector<double>( rhs.weights );
        biasAttr = rhs.biasAttr;

        return *this;
    }

	// Train the model to predict the labels
	void train(Matrix& features, Matrix& labels);

	// Evaluate the features and predict the labels
	void predict(const std::vector<double>& features, std::vector<double>& labels);
    
    // Activation function
    double activation(const std::vector<double>& feature, const double biasAttr, std::vector<double>& weights, const bool threshold = true);

    // Adjust the given weights and bias weight according to the perceptron rule
    void perceptronRule(const std::vector<double>& input, const double biasAttr, std::vector<double>& weights, const double target, const double output);

};

#endif // PERCEPTRON_H
