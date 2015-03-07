
#include "backprop.h"


void Backprop::train(Matrix& features, Matrix& labels)
{
    // get number of inputs
    size_t numInputs = features.cols();

    // reset max accuracy
    this->maxAccuracy = 0.0;
    this->maxCount = 0;
    

    // if hiddenNodes not initialized, make it 2 * numInputs
    // per the project spec
    if (!this->hiddenNodes < 0)
        this->hiddenNodes = 2 * numInputs;

    // set number of outputs for continuous/binary vs higher order outputs
    size_t numOutputs = labels.valueCount(0);
    this->continuousOut = false;
    if (numOutputs == 0)
        this->continuousOut = true;
    numOutputs = numOutputs < 3 ? 1 : numOutputs;

    // initialize weight vectors
    this->initWeights(numInputs, numOutputs);

    // initialize output vectors
    this->initOutputs(numInputs, numOutputs);
    
    // initialize error vectors
    this->initErrors(numInputs, numOutputs);

    // initialize validation set
    features.shuffleRows(m_rand, &labels);
    Matrix validation;
    Matrix validationLabels;
    double splitPercent = 0.75;
    this->splitValidationSet(features, labels, validation, validationLabels, splitPercent);

    std::vector< std::vector< std::vector<double> > > maxWeights;
    int bestEpoch = 1;
    int epoch = 0;
    double stopCriteria;
    size_t numFeatures = features.rows();
    bool stop;
//    std::cout << std::endl << "epoch,ClassAcc,MSE(TrS),MSE(VS)" << std::endl;
//    std::cout << epoch << ",";
//    std::cout << this->measureAccuracy(validation, validationLabels) << ",";
//    std::cout << this->getMeanSquaredError(features, labels) << ",";
//    std::cout << this->getMeanSquaredError(validation, validationLabels) << std::endl;
    do
    {
        ++epoch;

        // Shuffle the rows
        features.shuffleRows(m_rand, &labels);

        // for each feature
        for (size_t featureIndex = 0; featureIndex < numFeatures; ++featureIndex)
        {
            std::vector<double>& feature = features.row(featureIndex);
            // set input layer outputs to feature vector plus bias node
            std::vector<double> inputs = feature;
            inputs.push_back (1.0);
            this->outputs[0] = inputs;
            // run forward algorithm to calculate node outputs
            this->forward(this->weights, this->outputs);
            // run backprop algorithm to adjust node weights
            this->backward(this->weights, this->outputs, this->errors, labels.row(featureIndex)[0]);
        }

        // Get MSE over validation set
        stopCriteria = this->measureAccuracy(validation, validationLabels);
        double beforeAccuracy = this->maxAccuracy;
        stop = this->stop(stopCriteria);
        if (this->maxAccuracy > beforeAccuracy)
        {
            maxWeights = this->weights;
            bestEpoch = epoch;
        }

//        if (epoch % 5 == 0)
//        {
//            std::cout << epoch << ",";
//            std::cout << this->measureAccuracy(validation, validationLabels) << ",";
//            std::cout << this->getMeanSquaredError(features, labels) << ",";
//            std::cout << this->getMeanSquaredError(validation, validationLabels) << std::endl;
//        }

//        std::cout << "criteria: " << stopCriteria << std::endl;
//        std::cout << "max acc: " << this->maxAccuracy << std::endl;
//        std::cout << "epoch: " << epoch << std::endl;
    } while (epoch <= this->maxEpochs && !stop);// && this->maxAccuracy < 0.95);
    std::cout << std::endl;
    this->weights = maxWeights;
    /*for (size_t i = 0; i < this->weights.size(); ++i)
    {
        for (size_t j = 0; j < this->weights[i].size(); ++j)
        {
            for (size_t k = 0; k < this->weights[i][j].size(); ++k)
            {
                std::cout << " " << this->weights[i][j][k];
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }*/
    std::cout << "Best Epoch\n" << bestEpoch << std::endl << std::endl;
    std::cout << "Maximum Accuracy\n" << this->maxAccuracy << std::endl << std::endl;
    std::cout << "Training Set MSE\n" << this->getMeanSquaredError(features, labels) << std::endl << std::endl;
    std::cout << "Validation Set MSE\n" << this->getMeanSquaredError(validation, validationLabels) << std::endl;

}


void Backprop::predict(const std::vector<double>& features, std::vector<double>& labels)
{
    double MSE = 0.0;
    this->predict(features, labels, MSE);
}


void Backprop::predict(const std::vector<double>& features, std::vector<double>& labels, double& MSE)
{
    // set input layer outputs to feature vector
    std::vector<double> inputs = features;
    inputs.push_back (1.0);
    this->outputs[0] = inputs;
    // run forward algorithm to calculate node outputs
    this->forward(this->weights, this->outputs);

    size_t outputCount = this->outputs[this->outputIndex()].size();
    if (outputCount == 1)
    {
        double out = this->outputs[this->outputIndex()][0];
        if (!this->continuousOut)
            labels[0] = out < 0.0 ? 1.0 : 0.0;
        else
            labels[0] = out;
    }
    else
    {
        size_t maxLabel;
        double maxPred;
        (*((long long*)&maxPred)) = ~(1LL<<52); // awesome min double value thing
        double sse = 0.0;
        for (size_t i = 0; i < outputCount; ++i)
        {
            double target = 0.0;
            if (i == labels[0])
                target = 1.0;
//            std::cout << "outputIndex " << i << " output " << this->outputs[this->outputIndex()][i] << std::endl;
            double output = this->outputs[this->outputIndex()][i];
            if (maxPred < output)
            {
                maxPred = output;
                maxLabel = i;
            }
            double error = target - output;
            sse += error * error;
        }
        MSE += sse / outputCount;
        labels[0] = maxLabel;
    }

}

void Backprop::forward(const std::vector< std::vector< std::vector<double> > >& weights, std::vector< std::vector<double> >& outputs)
{
    // Check layer number assertion
    if(outputs.size() < 2)
        ThrowError("Backprop::forward:Expected there to be at least one input layer and one output layer");
    if(weights.size() != outputs.size())
        ThrowError("Backprop::forward:Expected the weights (minus output layer) and outputs to have the same number of layers");

    // Check layer size assertion
    size_t numLayers = this->numLayers();
    for (size_t layerIndex = 0; layerIndex < numLayers; ++layerIndex)
    {
        if (layerIndex < numLayers - 1 && outputs[layerIndex].size() < 2)
            ThrowError("Backprop::forward:Expected at least one regular node and one bias node in input and hidden layers");
        if (layerIndex == numLayers - 1 && outputs[layerIndex].size() < 1)
            ThrowError("Backprop::forward:Expected at least one node in output layer");
        if (layerIndex < numLayers - 1 && weights[layerIndex].size() != outputs[layerIndex].size())
            ThrowError("Backprop::forward:Expected the layers in weights and outputs to have the same number of nodes");
    }

    // for each hidden layer
    size_t inputLayer = 0;
    size_t biasCorrection = -1;
    for (size_t layerIndex = inputLayer + 1; layerIndex < numLayers; ++layerIndex)
    {
        if (layerIndex == numLayers - 1)
            biasCorrection = 0;
        else
            biasCorrection = -1; // this line should actually never execute, but it makes my intention more clear

        // get weights and outputs for previous layer to calculate nets
        std::vector< std::vector<double> > prevLayerWeights = weights[layerIndex - 1];
        std::vector<double>& prevLayerOutputs = outputs[layerIndex - 1];
        size_t prevNumNodes = prevLayerWeights.size();

        // get output vector for this layer so we can change it
        std::vector<double>& layerOutputs = outputs[layerIndex];
        size_t numNodes = layerOutputs.size(); 

        // for all regular nodes (exclude bias in non-output layers)
        for (size_t j = 0; j < numNodes + biasCorrection; ++j)
        {
            // compute net input from previous layer
            double net = 0.0;
            for (size_t i = 0; i < prevNumNodes; ++i)
                net += prevLayerWeights[i][j] * prevLayerOutputs[i];
            net *= -1;

            // compute output using sigmoid function
            layerOutputs[j] = 1 / (1 + exp (net));
        }
    }
}


void Backprop::backward(std::vector< std::vector< std::vector<double> > >& weights, const std::vector< std::vector<double> >& outputs, std::vector< std::vector<double> >& errors, const double& target)
{
    // Check layer number assertion
    if(outputs.size() < 2)
        ThrowError("Backprop::backward:Expected there to be at least one input layer and one output layer");
    if(weights.size() != outputs.size() || outputs.size() != errors.size())
        ThrowError("Backprop::backward:Expected the weights, outputs, and errors to have the same number of layers");

    // Check layer size assertion
    size_t numLayers = this->numLayers();
    for (size_t layerIndex = 0; layerIndex < numLayers; ++layerIndex)
    {
        if (layerIndex < numLayers - 1)
        {
            if (weights[layerIndex].size() < 2)
                ThrowError("Backprop::backward:Expected at least one regular node and one bias node in input and hidden layers of weights vector");
            if (weights[layerIndex].size() != outputs[layerIndex].size())
                ThrowError("Backprop::backward:Expected the layers in weights and outputs vectors to have the same number of nodes");
            if (layerIndex > 0 && errors[layerIndex].size() != outputs[layerIndex].size() - 1)
                ThrowError("Backprop::backward:Expected same number of non-bias nodes in outputs and errors vectors");
        }
        if (layerIndex == numLayers - 1)
        {
            if (outputs[layerIndex].size() < 1)
                ThrowError("Backprop::backward:Expected at least one node in output layer");
            if (errors[layerIndex].size() != outputs[layerIndex].size())
                ThrowError("Backprop::backward:Expected same number of output nodes in outputs and errors vectors");
            if (!this->continuousOut && target >= outputs[layerIndex].size())
                ThrowError("Backprop::backward:Expected target value to be a nominal within the output node range");
        }
    }

    // calculate error for hidden and output nodes
    for (size_t layerIndex = numLayers - 1; layerIndex > 0; --layerIndex)
    {
        std::vector<double>& errorVector = errors[layerIndex];
        std::vector<double> outputVector = outputs[layerIndex];
        for (size_t nodeIndex = 0; nodeIndex < outputVector.size(); ++nodeIndex)
        {
            double error = 1.0;
            if (layerIndex == numLayers - 1)
            {
                //std::cout << "nINdex " << nodeIndex << " bTarget " << target << std::endl;
                double nodeTarget = target;
                if (!this->continuousOut)
                {
                    if (nodeIndex == target)
                        nodeTarget = 1.0;
                    else
                        nodeTarget = 0.0;
                }
                error = this->calculateOutputError(nodeTarget, outputVector[nodeIndex]);
            }
            else
            {
                error = this->calculateHiddenError(weights[layerIndex][nodeIndex], errors[layerIndex + 1], outputVector[nodeIndex]);
            }
            errorVector[nodeIndex] = error;
        }
    }

    for (int layerIndex = numLayers - 2; layerIndex >= 0; --layerIndex)
    {
        std::vector< std::vector<double> >& weightVector = weights[layerIndex];
        std::vector<double> outputVector = outputs[layerIndex];
        std::vector<double> errorVector = errors[layerIndex + 1];
        for (size_t nextIndex = 0; nextIndex < errorVector.size(); ++nextIndex)
        {
            for (size_t thisIndex = 0; thisIndex < weightVector.size(); ++thisIndex)
            {
                std::vector<double>& nodeWeights = weightVector[thisIndex];
//                std::cout << "weight before " << nodeWeights[nextIndex] << " delta " << this->deltaRule(errorVector[nextIndex], outputVector[thisIndex]) << std::endl;
                double prevDelta = this->lastDelta[layerIndex][thisIndex][nextIndex];
                double thisDelta = this->deltaRule(errorVector[nextIndex], outputVector[thisIndex], prevDelta);
                nodeWeights[nextIndex] += thisDelta;
                this->lastDelta[layerIndex][thisIndex][nextIndex] = thisDelta;
//                std::cout << "next error " << errorVector[nextIndex] << " this output " << outputVector[thisIndex] << std::endl;
//                std::cout << "weight after " << weights[layerIndex][thisIndex][nextIndex] << std::endl;
            }
        }
    }
}


void Backprop::initWeights(const size_t& numInputs, const size_t& numOutputs)
{
    // get number of layers total
    size_t numLayers = this->numLayers();

    // set standard deviation for weight initialization
    double stdDev = 0.10;

    // add layers
    for (size_t layerIndex = 0; layerIndex < numLayers; ++layerIndex)
    {
        this->weights.push_back (std::vector< std::vector<double> > ());
        this->lastDelta.push_back (std::vector< std::vector<double> > ());
    }

    // add input layer weight vectors
    for (size_t nodeIndex = 0; nodeIndex < numInputs + 1; ++nodeIndex)
    {
        this->weights[0].push_back (std::vector<double> ());
        this->lastDelta[0].push_back (std::vector<double> ());
    }

    // for all hidden layers
    for (size_t layerIndex = 1; layerIndex < this->hiddenLayers + 1; ++layerIndex)
    {
        std::vector< std::vector<double> >& prevLayer = this->weights[layerIndex - 1];
        std::vector< std::vector<double> >& prevDeltaLayer = this->lastDelta[layerIndex - 1];
        for (size_t nodeIndex = 0; nodeIndex < this->hiddenNodes; ++nodeIndex)
        {
            std::vector< std::vector<double> >::iterator dIt = prevDeltaLayer.begin();
            // initialize weights from previous layer to this node
            for (std::vector< std::vector<double> >::iterator it = prevLayer.begin(); it != prevLayer.end(); ++it)
            {
                double initWeight = this->m_rand.normal() * stdDev;
                it->push_back (initWeight);
                dIt->push_back (0.0);
                std::advance (dIt, 1);
            }
            // add hidden layer weight vector
            this->weights[layerIndex].push_back (std::vector<double> ());
            this->lastDelta[layerIndex].push_back (std::vector<double> ());
        }
        // add bias node weight vector to hidden layer
        this->weights[layerIndex].push_back (std::vector<double> ());
        this->lastDelta[layerIndex].push_back (std::vector<double> ());
    }

    // for all output nodes
    for (size_t outputNode = 0; outputNode < numOutputs; ++outputNode)
    {
        // initialize weights from last layer to this output node
        std::vector< std::vector<double> >& lastLayer = this->weights[this->outputIndex() - 1];
        std::vector< std::vector<double> >& lastDeltaLayer = this->lastDelta[this->outputIndex() - 1];
        std::vector< std::vector<double> >::iterator dIt = lastDeltaLayer.begin();
        for (std::vector< std::vector<double> >::iterator it = lastLayer.begin(); it != lastLayer.end(); ++it)
        {
            double initWeight = this->m_rand.normal() * stdDev;
            it->push_back (initWeight);
            dIt->push_back (0.0);
            std::advance (dIt, 1);
        }
    }
}


void Backprop::initOutputs(const size_t& numInputs, const size_t& numOutputs)
{
    // get number of layers total
    size_t numLayers = this->numLayers();

    // add layers
    for (size_t layerIndex = 0; layerIndex < numLayers; ++layerIndex)
        this->outputs.push_back (std::vector<double> ());

    // init input node outputs
    for (size_t nodeIndex = 0; nodeIndex < numInputs; ++nodeIndex)
        this->outputs[0].push_back (0.0);
    this->outputs[0].push_back (1.0);

    // init hidden node outputs
    for (size_t layerIndex = 1; layerIndex < this->hiddenLayers + 1; ++layerIndex)
    {
        std::vector<double>& hiddenLayer = this->outputs[layerIndex];
        for (size_t nodeIndex = 0; nodeIndex < this->hiddenNodes; ++nodeIndex)
            hiddenLayer.push_back (0.0);
        hiddenLayer.push_back (1.0);
    }

    // init output node outputs
    for (size_t nodeIndex = 0; nodeIndex < numOutputs; ++nodeIndex)
        this->outputs[this->outputs.size() - 1].push_back (0.0);

}


void Backprop::initErrors(const size_t& numInputs, const size_t& numOutputs)
{
    // get number of layers total
    size_t numLayers = this->numLayers();

    // add layers
    for (size_t layerIndex = 0; layerIndex < numLayers; ++layerIndex)
        this->errors.push_back (std::vector<double> ());

    // init hidden node errors
    for (size_t layerIndex = 1; layerIndex < this->hiddenLayers + 1; ++layerIndex)
    {
        std::vector<double>& hiddenLayer = this->errors[layerIndex];
        for (size_t nodeIndex = 0; nodeIndex < this->hiddenNodes; ++nodeIndex)
            hiddenLayer.push_back (0.0);
    }

    // init output node errors
    for (size_t nodeIndex = 0; nodeIndex < numOutputs; ++nodeIndex)
        this->errors[this->outputIndex()].push_back (0.0);

}


double Backprop::calculateOutputError(const double& target, const double& output)
{
//    std::cout << "target " << target << " output " << output << " error " << (target - output) * output * (1 - output) << std::endl;
    return (target - output) * output * (1 - output);
}


double Backprop::calculateHiddenError(const std::vector<double>& layerWeights, const std::vector<double>& nextLayerErrors, const double& output)
{
    if (layerWeights.size() != nextLayerErrors.size())
        ThrowError("Expected weights and errors to have same number of values");
    double error = 0.0;
    for (size_t index = 0; index < layerWeights.size(); ++index)
    {
        error += layerWeights[index] * nextLayerErrors[index];
    }
    error *= output * (1 - output);
    return error;
}


double Backprop::deltaRule(const double& toError, const double& output, const double& prevDelta)
{
    return (this->learningRate * toError * output) + (this->momentum * prevDelta);
}


size_t Backprop::numLayers()
{
    if (this->hiddenNodes == 0)
        this->hiddenLayers = 0;
    return 1 + this->hiddenLayers + 1;
}


size_t Backprop::outputIndex()
{
    return this->hiddenLayers + 1;
}


bool Backprop::stop(const double& stopCriteria)
{
    double accuracy = stopCriteria;
    if (this->continuousOut) // then stopCriteria is actually MSE
        accuracy = 1 - accuracy;

    if (accuracy > this->maxAccuracy)
    {
        this->maxAccuracy = accuracy;
        this->maxCount = 0;
    }
    else
        ++this->maxCount;

    if (this->maxCount > this->maxEpochs / 10)
        return true;
    return false;
}


double Backprop::getMeanSquaredError(Matrix& features, Matrix& labels)
{
    double MSE = 0.0;
    std::vector<double> pred;
    pred.resize(1);
    for(size_t i = 0; i < features.rows(); i++)
    {
        const std::vector<double>& feat = features.row(i);
        pred[0] = labels.row(i)[0];
        predict(feat, pred, MSE);
    }
    return MSE / features.rows();
}

