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

    virtual double dist(const std::vector<double>&, const std::vector<double>&);

private:

    Rand m_rand;

    size_t k;

    double replaceTop(std::vector<RowDistance>&, size_t, double);

    double vote(const std::vector<RowDistance>&, bool weight = false);

protected:

    Matrix features;
    Matrix labels;

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

    virtual double dist(const std::vector<double>&, const std::vector<double>&);

private:

    std::map<double, size_t> labelValueCounts;

	size_t m_bins;
	std::vector<double> m_featureMins;
	std::vector<double> m_featureMaxs;
    std::vector<double> m_featureWidths;

    std::map<size_t, std::map<double, std::map<double, double> > > probabilities;

    void trainFilter(Matrix&);
    Matrix discretizeFeatures(Matrix&);
    
    // Based off of equation (18) from http://axon.cs.byu.edu/~martinez/classes/478/readings/Wilson_distance.pdf
    std::vector<double> discretize(const std::vector<double>&);

    // Based off of part 3 of equation (18) from http://axon.cs.byu.edu/~martinez/classes/478/readings/Wilson_distance.pdf
    size_t getBin(double, double, double);

    // Based off of equation (23) from http://axon.cs.byu.edu/~martinez/classes/478/readings/Wilson_distance.pdf
    double calcPac(double, size_t, size_t, double);

    // Based off of equation (24) from http://axon.cs.byu.edu/~martinez/classes/478/readings/Wilson_distance.pdf
    double calcMid(size_t, size_t);

};

#endif // KNN_H
