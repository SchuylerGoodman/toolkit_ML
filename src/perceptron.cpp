
#include "perceptron.h"


void Perceptron::train(Matrix& features, Matrix& labels)
{
    // Check assumptions
    if(features.rows() != labels.rows())
        ThrowError("Expected the features and labels to have the same number of rows");

    // save the number of inputs
    int nInputs = (int)features.rows();
    
    // save the number of attributes
    int nAttrs = (int)features.cols();

    // initialize the weights
    // TODO think about what values I should initialize these to (small random?)
    this->weights.resize(nAttrs + 1, 0.0);

    this->biasAttr = 1.0;
    int epochs = 0;
    int wrongs = 0;
    double maxAcc = -1.0;
    std::vector<double> maxWeights;
    int sinceMax = 0;

    // loop through the inputs until analysis end
    do
    {
        wrongs = 0;

        // Shuffle the rows
        features.shuffleRows(m_rand, &labels);

        // for every input vector:
        //  compute activation function and,
        //  adjust weights
        for (int featureIndex = 0; featureIndex < nInputs; ++featureIndex)
        {
            std::vector<double> feature = features.row(featureIndex);
            if ((int)feature.size() != nAttrs)
                ThrowError("Expected the feature to have the same number of attributes");

            // compute activation
            double target = labels.row(featureIndex)[0];
            double output = this->activation(feature, this->biasAttr, this->weights);
//            std::cout << "targ " << target << " out " << output << " diff " << target - output << std::endl;
            if (target - output != 0.0) // if one of the inputs is incorrect
            {
                ++wrongs;
            }

            this->perceptronRule(feature, this->biasAttr, this->weights, target, output);
        }

        double accuracy = this->measureAccuracy(features, labels);
//        std::cout << "acc vs maxAcc " << accuracy << " " << maxAcc << std::endl;
        if (accuracy > maxAcc)
        {
            maxAcc = accuracy;
//            std::cout << "max acc " << maxAcc << std::endl;
            maxWeights = this->weights;
            sinceMax = 0;
        }
        else
            ++sinceMax;

        ++epochs;

    } while (wrongs > 0 && sinceMax < (this->maxEpochs / 10));
    std::cout << "Epochs completed: " << std::endl << epochs << std::endl;
    this->weights = maxWeights;

    std::cout << "Final weights: " << std::endl;
    for (size_t i = 0; i < weights.size(); ++i)
    {
        if (i > 0)
            std::cout << ",";
        std::cout << weights[i];
    }
    std::cout << std::endl << std::endl;

}


void Perceptron::predict(const std::vector<double>& features, std::vector<double>& labels)
{
    labels[0] = this->activation(features, this->biasAttr, this->weights, true);
}


double Perceptron::activation(const std::vector<double>& feature, const double biasAttr, std::vector<double>& weights, const bool threshold)
{
    double activation = 0.0;
    int featureSize = feature.size();

    // sum weights
    for (int attrIndex = 0; attrIndex < featureSize; ++attrIndex)
    {
        double value = feature[attrIndex];
        double weight = weights[attrIndex];
        activation += value * weight;
    }
    // add bias to activation function
    activation += biasAttr * weights[featureSize];

    // threshold activation function
    if (threshold)
        return activation > 0.0 ? 1.0 : 0.0;
    return activation;
}


void Perceptron::perceptronRule(const std::vector<double>& input, const double biasAttr, std::vector<double>& weights, const double target, const double output)
{
    double t_output = output > 0.0 ? 1.0 : 0.0;
    // compute part of perceptron rule
    double diff = this->learningRate * (target - t_output);

    int inputSize = input.size();
    // adjust weights
    for (int attrIndex = 0; attrIndex < inputSize; ++attrIndex)
    {
        double value = input[attrIndex];
        // learn weights by perceptron rule
        double dWeight = diff * value;
        weights[attrIndex] += dWeight;
    }
    // do the same for bias weight
    weights[inputSize] += diff * biasAttr;
}

