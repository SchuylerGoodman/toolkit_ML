
#include "treenode.h"

// initialize static member - this is a hack because I don't want to traverse the tree to get it
// I could make a shared_ptr<size_t> to hold it TODO
size_t TreeNode::MAX_DEPTH = 0;

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
    newNode->setAttr(-1, "");
    newNode->setValue(-1, "");
    newNode->setLabel(-1, "");
    newNode->depth = this->depth + 1;
    //std::cout << this->depth << " " << MAX_DEPTH << " " << newNode->depth << std::endl;
    if (newNode->depth > MAX_DEPTH)
        MAX_DEPTH = newNode->depth;

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


void TreeNode::printTree(NodePtr node)
{
    std::string space = "   |";

    for (size_t i = 0; i < node->getDepth(); ++i)
        std::cout << space;

    std::cout << " " << node->getValueName() << " - ";

    if (node->getAttr() >= 0)
        std::cout << node->getAttrName();

    if (node->isLeaf())
        std::cout << "***" << node->getLabelName() << "***";
    
    std::cout << std::endl;

    for (TreeNode::iterator it = node->begin(); it != node->end(); ++it)
        printTree((*it));
}
