
#ifndef NBPERCEPTRON_H
#define NBPERCEPTRON_H

#include "perceptron.h"
#include "matrix.h"
#include "rand.h"
#include "error.h"
#include <vector>

class NBPerceptron : public SupervisedLearner
{
private:
    Rand& m_rand;
    int maxEpochs;
    double learningRate;
    
    std::vector<Perceptron> perceptrons;
        
    static const int MAX_EPOCHS = 300;
    static const double LEARNING_RATE = 0.1;

public:
    NBPerceptron(Rand& r, int maxEpochs = MAX_EPOCHS, double learningRate = LEARNING_RATE)
    : SupervisedLearner(), m_rand(r), maxEpochs(maxEpochs), learningRate(learningRate)
    {
    }

    ~NBPerceptron()
    {
    }

	// Train the model to predict the labels
	void train(Matrix& features, Matrix& labels);

	// Evaluate the features and predict the labels
	void predict(const std::vector<double>& features, std::vector<double>& labels);
    
    // Use a hard max function to pick the predicted label
    void hardMax(const std::vector<double>& predictions, std::vector<double>& labels);

    // Normalize the continuous values in the features matrix between min and max
    void normalizeFeatures(Matrix& features, const double min, const double max);
};

#endif // NBPERCEPTRON_H
