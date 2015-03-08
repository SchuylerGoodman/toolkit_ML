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
        : labeled(false), label(-1), labelName(""),
        attr(-1), attrName(""),
        value(-1), valueName(""), depth(1)
    {}

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

    // Gets the maximum depth below and including this
    const size_t getMaxDepth();

    // Gets the number of nodes below and including this
    const size_t getNodeCount();
    
    //++++++++++++++++++++++public methods++++++++++++++++++++++

    // Adds a new NodePtr to the list of children and returns it
    NodePtr add();

    // Factory function for new NodePtr objects
    static NodePtr make();

    // Factory deep copy method for NodePtr objects
    static NodePtr copy(NodePtr);

    // Checks if this node has any child nodes
    bool isLeaf() { return children.empty(); };

    // Disables the node - makes it a leaf node
    void disable();

    // Enables the node - reattaches the node's children
    void enable();

    // Prints the tree to std out
    static void printTree(NodePtr);
    
private:

    std::vector< NodePtr > children;
    std::vector< NodePtr > disabled_children;

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
