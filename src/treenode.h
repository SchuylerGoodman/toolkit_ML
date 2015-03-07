#ifndef TREENODE_H
#define TREENODE_H

#include <memory>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>
#include <vector>
#include <iostream>

#include "matrix.h"

class TreeNode
{
public:

    TreeNode()
        : labeled(false)
    {}

    // There's something funky going on with depth in this constructor TODO
    TreeNode(const TreeNode& other)
        : labeled(other.labeled), label(other.label), labelName(other.labelName),
        labels(other.labels), attr(other.attr), attrName(other.attrName),
        value(other.value), valueName(other.valueName), depth(other.depth)
    {
        for (TreeNode::const_iterator it = other.begin(); it != other.end(); ++it)
        {
            children.push_back(TreeNode::copy(*it));
        }
    }

    // boost pointer typedef
    typedef boost::shared_ptr<TreeNode> NodePtr;

    // iterator definitions
    typedef std::vector< NodePtr >::iterator iterator;
    typedef std::vector< NodePtr >::const_iterator const_iterator;
    iterator begin() { return children.begin(); };
    const_iterator begin() const { return children.begin(); };
    iterator end() { return children.end(); };
    const_iterator end() const { return children.end(); };

    //+++++++++++++++++++getters and setters++++++++++++++++++++

    void setLabel(double, std::string);
    const double getLabel();
    const std::string getLabelName();
    const bool isLabeled() { return this->labeled; };

    void setLabels(Matrix&);

    void setAttr(size_t, std::string);
    const size_t getAttr();
    const std::string getAttrName();

    void setValue(double, std::string);
    const double getValue();
    const std::string getValueName();

    const size_t getDepth();
    
    //++++++++++++++++++++++public methods++++++++++++++++++++++

    // Adds a new NodePtr to the list of children and returns it
    NodePtr add();

    // Factory function for new NodePtr objects
    static NodePtr make();

    // Factory copy method for NodePtr objects
    static NodePtr copy(NodePtr);

    // Checks if this node has any child nodes
    bool isLeaf() { return children.empty(); };

    // Prints the tree to std out
    static void printTree(NodePtr);
    
private:

    static size_t MAX_DEPTH;

    std::vector< NodePtr > children;

    bool labeled;
    double label;
    std::string labelName;

    Matrix labels;

    size_t attr;
    std::string attrName;

    double value;
    std::string valueName;

    size_t depth;

};

#endif // TREENODE_H
