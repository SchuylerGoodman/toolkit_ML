
#include "nbperceptron.h"

void NBPerceptron::train(Matrix& features, Matrix& labels)
{
    // get number of output values
    size_t valueCount = labels.valueCount(0);
//    this->normalizeFeatures(features, -1.0, 1.0);
//    features.toCSV();

    Matrix classLabels ( labels );
    Matrix classFeatures ( features );
    for (size_t i = 0; i < valueCount; ++i)
    {
        // create a perceptron for each output value
        this->perceptrons.push_back(Perceptron( this->m_rand, this->maxEpochs, this->learningRate, false ));

        // copy the features matrix for each perceptron (because of shuffling rows)
        classFeatures.setSize(0, 0);
        classFeatures.copyPart(features, 0, 0, features.rows(), features.cols());

        // copy the labels matrix for each output value
        // also, alter the matrix so only outputs corresponding to the i-th perceptron are 1 (all others are 0)
        classLabels.setSize(0, 0);
        classLabels.copyPart(labels, 0, 0, labels.rows(), labels.cols());
        for (size_t j = 0; j < classLabels.rows(); ++j)
        {
//            std::cout << j << "," << classLabels.row(j)[0] << ", " << i << std::endl;
            if (classLabels.row(j)[0] == i) // 1 for every value corresponding to the i-th perceptron
                classLabels.row(j)[0] = 1;
            else // otherwise, 0
                classLabels.row(j)[0] = 0;
        }

//        classLabels.toCSV();
        this->perceptrons[i].train(classFeatures, classLabels);
    }
}


void NBPerceptron::predict(const std::vector<double>& features, std::vector<double>& labels)
{
    size_t valueCount = this->perceptrons.size();
    std::vector<double> preds;
    preds.resize(valueCount, 0.0);

    std::vector<double> classLabels;
    classLabels.resize(1);
    for (size_t i = 0; i < valueCount; ++i)
    {
        this->perceptrons[i].predict(features, classLabels);
        preds[i] = classLabels[0];
//        std::cout << "true: " << i << " pred: " << classLabels[0] << std::endl;
    }

    this->hardMax(preds, labels);
}


void NBPerceptron::hardMax(const std::vector<double>& predictions, std::vector<double>& labels)
{
    size_t predCount = predictions.size();
    size_t maxLabel;
    double maxPred;
    (*((long long*)&maxPred)) = ~(1LL<<52); // awesome min double value thing
    for (size_t i = 0; i < predCount; ++i)
    {
        if (maxPred < predictions[i])
        {
            maxPred = predictions[i];
            maxLabel = i;
        }
    }
//    std::cout << "max pred " << maxPred << " label " << maxLabel << std::endl;
    labels[0] = maxLabel;
}


void NBPerceptron::normalizeFeatures(Matrix& features, const double min, const double max)
{
    double newRange = max - min;
    for (size_t i = 0; i < features.cols(); ++i)
    {
        if (features.valueCount(i) != 0)
            continue;

        double colMin = features.columnMin(i);
        double colMax = features.columnMax(i);
        double oldRange = colMax - colMin;

        for (size_t j = 0; j < features.rows(); ++j)
        {
            double val = features[j][i];
            // scale val between min and max
            features[j][i] = ( ( val - colMin ) / oldRange ) * newRange + min;
//            std::cout << "min " << colMin << " max " << colMax << " old " << val << " new " << features[j][i] << std::endl;
        }
    }
}
    
