
#include "decisiontree.h"


void DecisionTree::train(Matrix& features, Matrix& labels)
{
    if (root)
        root.reset();

    root = TreeNode::make();
    root->setValue(0, "root");

    // shuffle instances and use unknown nominal values
    features.shuffleRows(m_rand, &labels);
    features.useUnknown();

    // split validation set
    Matrix valFeatures;
    Matrix valLabels;
    double splitPercent = 0.75;
    this->splitValidationSet(features, labels, valFeatures, valLabels, splitPercent);

    // begin recursion
    this->partition(root, features, labels);
    //TreeNode::printTree(root);
    //std::cout << "Max depth," << root->getMaxDepth() << std::endl;
    //std::cout << "Node count," << root->getNodeCount() << std::endl;
    //std::cout << "\n\n";

//    std::cout << "pruning" << std::endl;
//    this->prune(valFeatures, valLabels);
//    std::cout << "Max depth," << root->getMaxDepth() << std::endl;
//    std::cout << "Node count," << root->getNodeCount() << std::endl;
//    std::cout << "\n\n";
//    TreeNode::printTree(root);
    
}


void DecisionTree::predict(const std::vector<double>& features, std::vector<double>& labels)
{
    this->dive(root, features, labels);
}


void DecisionTree::dive(TreeNode::NodePtr node, const std::vector<double>& features, std::vector<double>& labels)
{
    // if the current node is a leaf node
    if (node->isLeaf())
    {   // set the label and return
        labels[0] = node->getLabel();
        return;
    }

    // get the current node's attribute
    size_t attr = node->getAttr();

    // copy and reduce the feature vector by the current attribute so the attr numbering doesn't get off
    std::vector<double> featureCopy = std::vector<double> (features);
    std::vector<double>::iterator position = featureCopy.begin() + attr;
    featureCopy.erase(position);

    // for each child node
    for (TreeNode::iterator it = node->begin(); it != node->end(); ++it)
    {
        double compare = (*it)->getValue();
        // if the child node value matches the given input feature, dive in
        if ((*it)->getValue() != features[attr] && (*it)->getValueName() == "?")
            compare = UNKNOWN_VALUE;
        if (compare == features[attr])
        {
            this->dive((*it), featureCopy, labels);
            return;
        }
    }
    std::cout << "attribute " << node->getAttr() << "-" << node->getAttrName() << " value " << features[attr] << std::endl;
    ThrowError("No matching attribute-value pair for input feature");
}


bool DecisionTree::partition(TreeNode::NodePtr node, Matrix& features, Matrix& labels)
{
    if (features.rows() != labels.rows())
        ThrowError("Partition::Feature and labels matrices should have the same number of nodes.");

    // save labels for later use
    node->setLabels(labels);

    size_t rows = features.rows();
    double dblrows = (double) rows;

    std::map<double, size_t> classes = labels.getValueCounts(0);
    size_t numClasses = classes.size();
    if (numClasses == 1)
    {
        // label node with class
        double c = classes.begin()->first;
        node->setLabel(c, labels.attrValue(0, c));
        return true;
    }
    else if (numClasses < 1 || rows < 1 || features.cols() < 1) // what if # remaining attributes == 1? Do we need to calc gain?
    {
        return false;
    }

    // otherwise, continue partitioning

    double entropy = this->calculateEntropy(features, labels);
    double maxGain = 0;
    size_t maxAttr = 0;

    // for each attribute
    for (size_t attr = 0; attr < features.cols(); ++attr)
    {
        std::map<double, size_t> counts = features.getValueCounts(attr);

        // if there is only one value, this attribute doesn't help at all. Just skip it.
        if (counts.size() == 1)
            continue;

        double info = 0.0;
        double gainRatio = 0.0;

        // for each value
        for (std::map<double, size_t>::iterator it = counts.begin(); it != counts.end(); ++it)
        {
            // set size_t value to invalid size unless it is valid
            size_t value = -1;
            if (it->first != UNKNOWN_VALUE)
                value = it->first;

            // get reduced matrices
            Matrix reducedFeatures;
            Matrix reducedLabels;
            std::vector<size_t> rowIndices = features.copyReduce(reducedFeatures, attr, value);
            labels.copyReduce(reducedLabels, rowIndices);

            // get rows in reduced matrix
            size_t redRows = reducedFeatures.rows();

            // get rows in reduced / rows in features
            double rowRatio = (double) redRows / dblrows;

            // calculate entropy for reduced matrices
            double attrEntropy = this->calculateEntropy(reducedFeatures, reducedLabels);

            // calculate info using entropy and row # division
            info += rowRatio * attrEntropy;

            // calculate gain ratio
            gainRatio -= rowRatio * log2 (rowRatio);
        }

        // calculate gain for value.
        double gain = entropy - info;
        //gain = gain / gainRatio;

        // if maximum, save attribute value
        if (gain > maxGain)
        {
            maxGain = gain;
            maxAttr = attr;
        }
    }

    // label node with best attribute
    node->setAttr(maxAttr, features.attrName(maxAttr));

    // for each value in max attribute
    std::vector<size_t> values = features.allAttrValues(maxAttr);
    for (std::vector<size_t>::iterator it = values.begin(); it != values.end(); ++it)
    {
        size_t value = (*it);
        //std::cout << value << "-" << features.attrValue(maxAttr, value) << std::endl;

        // create a new node in node
        TreeNode::NodePtr newNode = node->add();
        newNode->setValue(value, features.attrValue(maxAttr, value));

        // get reduced matrices (again?)
        Matrix reducedFeatures;
        Matrix reducedLabels;
        std::vector<size_t> rowIndices = features.copyReduce(reducedFeatures, maxAttr, value);
        labels.copyReduce(reducedLabels, rowIndices);

        // call partition with that node and reduced matrices
        bool labeled = this->partition(newNode, reducedFeatures, reducedLabels);
        if (!labeled) // then label new node with most common label
        {
            double mcl = labels.mostCommonValue(0);
            newNode->setLabel(mcl, labels.attrValue(0, mcl));
        }
    }

    return true;
}


void DecisionTree::prune(Matrix& valFeatures, Matrix& valLabels)
{
    PruneData pruneData;

    // initialize error
	size_t labelValues = valLabels.valueCount(0);
    double error = measureAccuracy(valFeatures, valLabels);
    if (labelValues != 0)
        error = 1 - error;
    pruneData.error = error;
    
    // Until the pruning does not reduce error
    while (this->prune(root, valFeatures, valLabels, pruneData))
    {
        // disable nodes that reduce error the most
        //std::cout << "disabling attr " << pruneData.pruned->getAttrName() << " value " << pruneData.pruned->getValueName() << std::endl;
        pruneData.pruned->disable();
        pruneData.pruned.reset();
    }
}


bool DecisionTree::prune(TreeNode::NodePtr node, Matrix& valFeatures, Matrix& valLabels, PruneData& pruneData)
{
    bool pruned = false;

    if (!node || node->isLeaf())
        return pruned;

	size_t labelValues = valLabels.valueCount(0);

    for (TreeNode::iterator it = node->begin(); it != node->end(); ++it)
    {
        TreeNode::NodePtr child = (*it);

        // recurse to check leaves first
        if (this->prune(child, valFeatures, valLabels, pruneData))
            pruned = true;
        
        // disable the node to check if the error gets reduced
        node->disable();

        // measure error (if label is nominal we will get accuracy, so reverse it)
        double error = this->measureAccuracy(valFeatures, valLabels);
        if (labelValues != 0)
            error = 1 - error;

        // re-enable so we can test other prunings to find the best one
        node->enable();

        // if the error was reduced, this pruning is better
        if (error < pruneData.error)
        {
            //std::cout << "pruning - error was " << pruneData.error << " new pruning got " << error << std::endl;
            pruneData.pruned = node;
            pruneData.error = error;
            pruned = true;
        }
    }

    return pruned;
}


double DecisionTree::calculateEntropy(Matrix& features, Matrix& labels)
{
    std::map<double, size_t> counts = labels.getValueCounts(0);
    double numRows = labels.rows();

    double entropy = 0.0;
    for (std::map<double, size_t>::iterator it = counts.begin(); it != counts.end(); ++it)
    {
        double p = it->second / numRows;
        if (p)
            entropy += p * log2 (p);
    }
    entropy *= -1;

    return entropy;
}
