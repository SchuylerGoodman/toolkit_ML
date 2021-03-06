#ifndef DECISIONTREE_H
#define DECISIONTREE_H

#include <map>
#include <cmath>

#include "matrix.h"
#include "learner.h"
#include "rand.h"
#include "error.h"
#include "treenode.h"

class DecisionTree : public SupervisedLearner
{
public:

    DecisionTree(Rand r)
        : m_rand(r)
    {}

	// Train the model to predict the labels
	void train(Matrix&, Matrix&);

	// Evaluate the features and predict the labels
	void predict(const std::vector<double>&, std::vector<double>&);

    void dive(TreeNode::NodePtr node, const std::vector<double>& features, std::vector<double>& labels);

	bool partition(TreeNode::NodePtr, Matrix&, Matrix&);

    void prune(Matrix&, Matrix&);

private:

    struct PruneData {
        TreeNode::NodePtr pruned;
        double error;
    };

    Rand m_rand;
    
    TreeNode::NodePtr root;

    bool prune(TreeNode::NodePtr, Matrix&, Matrix&, PruneData&);

    double calculateEntropy(Matrix&, Matrix&);

};

#endif // DECISIONTREE_H
