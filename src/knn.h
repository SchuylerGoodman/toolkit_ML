#ifndef KNN_H
#define KNN_H


#include <cmath>
#include <utility>
#include <limits>
#include <algorithm>

#include "learner.h"
#include "rand.h"
#include "error.h"


typedef std::pair<size_t, double> RowDistance;


// Boring KNN learner
class KNN : public SupervisedLearner
{
public:
    KNN(Rand rand)
        : m_rand(rand)
    {}

    void train(Matrix&, Matrix&);

    void predict(const std::vector<double>&, std::vector<double>&);

private:

    Rand m_rand;

    size_t k;

    Matrix features;
    Matrix labels;

    double dist(const std::vector<double>&, const std::vector<double>&);

    double replaceTop(std::vector<RowDistance>&, size_t, double);

    double vote(const std::vector<RowDistance>&, bool weight = true);

};


// IVDM learner
class IVDM : public KNN
{
public:
    IVDM(Rand rand)
        : KNN(rand)
    {}

    void train(Matrix&, Matrix&);

    void predict(const std::vector<double>&, std::vector<double>&);

private:

	size_t m_bins;
	std::vector<double> m_featureMins;
	std::vector<double> m_featureMaxs;

    void trainFilter(Matrix&);
    Matrix discretizeFeatures(Matrix&);
    
    std::vector<double> discretize(const std::vector<double>&);

};

#endif // KNN_H
