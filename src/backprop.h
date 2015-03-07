#ifndef BACKPROP_H
#define BACKPROP_H


#include <cmath>
#include <iterator>
#include <vector>

#include "matrix.h"
#include "learner.h"
#include "rand.h"
#include "error.h"
#include "time.h"

// Implementation of a MLP using backpropagation to minimize MSE.
// Assumes every node in a layer is connected to every node in preceding and following layers
class Backprop : public SupervisedLearner
{
    Rand m_rand;
    int maxEpochs;
    double learningRate;
    double momentum;
    size_t hiddenLayers;
    size_t hiddenNodes;
    bool continuousOut;
    double maxAccuracy;
    int maxCount;

    std::vector< std::vector< std::vector<double> > > weights;
    std::vector< std::vector< std::vector<double> > > lastDelta;
    std::vector< std::vector<double> > outputs;
    std::vector< std::vector<double> > errors;
    double biasAttr;

    static const int MAX_EPOCHS = 1000;
    static const double LEARNING_RATE = 0.6;
    static const double MOMENTUM = 0.2;
    static const size_t HIDDEN_LAYERS = 4;

    // This will be set to twice the input size unless a positive value is given
    static const size_t HIDDEN_NODES = 32;

public:
    Backprop()
    : SupervisedLearner(), maxEpochs(MAX_EPOCHS), learningRate(LEARNING_RATE), momentum(MOMENTUM), 
      hiddenLayers(HIDDEN_LAYERS), hiddenNodes(HIDDEN_NODES)
    {
    }

    Backprop(Rand r, int maxEpochs = MAX_EPOCHS, double learningRate = LEARNING_RATE, double momentum = MOMENTUM,
                int hiddenLayers = HIDDEN_LAYERS, int hiddenNodes = HIDDEN_NODES)
    : SupervisedLearner(), m_rand(r), maxEpochs(maxEpochs), learningRate(learningRate), momentum(momentum), 
      hiddenLayers(hiddenLayers), hiddenNodes(hiddenNodes)
    {
    }

	~Backprop()
	{
	}

    Backprop(const Backprop& p)
    :   m_rand(p.m_rand), maxEpochs(p.maxEpochs), learningRate(p.learningRate),
        momentum(p.momentum), hiddenLayers(p.hiddenLayers), hiddenNodes(p.hiddenNodes), 
        weights(p.weights), biasAttr(p.biasAttr)
    {
    }
    
    Backprop& operator=(const Backprop& rhs)
    {
        m_rand = rhs.m_rand;
        maxEpochs = rhs.maxEpochs;
        learningRate = rhs.learningRate;
        momentum = rhs.momentum;
        hiddenLayers = rhs.hiddenLayers;
        hiddenNodes = rhs.hiddenNodes;
        
        weights.clear();
        for (size_t i = 0; i < rhs.weights.size(); ++i)
        {
            weights.push_back (std::vector< std::vector<double> > ());
            for (size_t j = 0; j < rhs.weights[i].size(); ++j)
            {
                weights[i].push_back (std::vector<double>( rhs.weights[i][j] ));
            }
        }
        biasAttr = rhs.biasAttr;

        return *this;
    }

	// Train the model to predict the labels
	void train(Matrix&, Matrix&);

	// Evaluate the features and predict the labels
	void predict(const std::vector<double>&, std::vector<double>&);
    // Predict overload, if a double is provided it will put the MSE there, if possible
    void predict(const std::vector<double>&, std::vector<double>&, double&);
    
    // Computes the outputs for all layers for a single feature vector
    // Assumes the first vector in the outputs parameter is initialized from feature vector
    // Size of vectors in weights and outputs must match
    void forward(const std::vector< std::vector< std::vector<double> > >&, std::vector< std::vector<double> >&);

    // Computes the errors for the output and hidden layers, going backwards (the backprop step)
    // Assumes the forward algorithm has been run to update the output vector
    void backward(std::vector< std::vector< std::vector<double> > >&, const std::vector< std::vector<double> >&, std::vector< std::vector<double> >&, const double&);

    double getMeanSquaredError(Matrix&, Matrix&);

private:

    // Initialize the weights vectors vectors vector victor
    // Want weights for input and hidden layers
    void initWeights(const size_t&, const size_t&);

    // Initialize the outputs vector
    // Want outputs for all layers
    void initOutputs(const size_t&, const size_t&);

    // Initialize the errors vector
    // Want errors for hidden (excluding bias nodes) and output layers
    void initErrors(const size_t&, const size_t&);

    // Calculate error for output node
    double calculateOutputError(const double&, const double&);

    // Calculate error for hidden node
    double calculateHiddenError(const std::vector<double>&, const std::vector<double>&, const double&);

    // Calculate delta weight
    double deltaRule(const double&, const double&, const double&);

    // Returns the number of layers in the MLP
    size_t numLayers();

    // Calculates the index of the output layer
    size_t outputIndex();

    // Stopping criteria
    bool stop(const double& stopCriteria);



};

#endif // BACKPROP_H
