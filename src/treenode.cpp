
#include "treenode.h"

TreeNode::NodePtr TreeNode::make()
{
    NodePtr newNode = boost::make_shared<TreeNode> ();
    return newNode;
}


TreeNode::NodePtr TreeNode::copy(NodePtr other)
{
    NodePtr copyNode = boost::make_shared<TreeNode> (*other);
    return copyNode;
}


TreeNode::NodePtr TreeNode::add()
{
    NodePtr newNode = TreeNode::make();
    newNode->depth = this->depth + 1;

    this->children.push_back(newNode);
    return newNode;
}


void TreeNode::setLabel(double label, std::string labelName)
{
    this->label = label;
    this->labelName = labelName;
    this->labeled = true;
}


const double TreeNode::getLabel()
{
    return this->label;
}


const std::string TreeNode::getLabelName()
{
    return this->labelName;
}


void TreeNode::setLabels(Matrix& labels)
{
    this->labels = Matrix(labels);
    this->labels.copyPart(labels, 0, 0, labels.rows(), labels.cols());
}


void TreeNode::setAttr(size_t attr, std::string attrName)
{
    this->attr = attr;
    this->attrName = attrName;
}


const size_t TreeNode::getAttr()
{
    return this->attr;
}


const std::string TreeNode::getAttrName()
{
    return this->attrName;
}


void TreeNode::setValue(double value, std::string valueName)
{
    this->value = value;
    this->valueName = valueName;
}


const double TreeNode::getValue()
{
    return this->value;
}


const std::string TreeNode::getValueName()
{
    return this->valueName;
}


const size_t TreeNode::getDepth()
{
    return this->depth;
}


const size_t TreeNode::getMaxDepth()
{
    size_t maxDepth = this->depth;
    for (TreeNode::iterator it = this->begin(); it != this->end(); it++)
    {
        size_t depth = (*it)->getMaxDepth();
        if (depth > maxDepth)
            maxDepth = depth;
    }
    return maxDepth;
}


const size_t TreeNode::getNodeCount()
{
    size_t count = 1;
    for (TreeNode::iterator it = this->begin(); it != this->end(); it++)
    {
        count += (*it)->getNodeCount();
    }
    return count;
}


void TreeNode::disable()
{
    for (TreeNode::iterator it = this->begin(); it != this->end(); it++)
    {
        this->disabled_children.push_back(*it);
    }
    this->children.erase(this->begin(), this->end());

    double mcl = this->labels.mostCommonValue(0);
    this->setLabel(mcl, this->labels.attrValue(0, mcl));
}


void TreeNode::enable()
{
    for (std::vector<NodePtr>::iterator it = disabled_children.begin(); it != disabled_children.end(); ++it)
    {
        this->children.push_back(*it);
    }
    this->disabled_children.erase(disabled_children.begin(), disabled_children.end());

    this->setLabel(-1, "");
}


void TreeNode::printTree(NodePtr node)
{
    std::string space = "   |";

    for (size_t i = 1; i < node->getDepth(); ++i)
        std::cout << space;

    std::cout << " " << node->getValueName() << " - ";

    if (node->getAttr() >= 0)
        std::cout << node->getAttrName();

    if (node->isLeaf())
        std::cout << " - ***" << node->getLabelName() << "***";
    
    std::cout << std::endl;

    for (TreeNode::iterator it = node->begin(); it != node->end(); ++it)
        printTree((*it));

    std::cout.flush();
}
