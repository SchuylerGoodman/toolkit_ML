#include "backprop.h"
#include "tests/include/gtest/gtest.h"

/*
 * i-+-h
 *  \|/ \
 *   X   o
 *  /|\ /|
 * i-+-h |
 *   //  /
 *  //  /
 *  /  /
 * b  b
 */
TEST(BackpropTest, forward)
{
    Rand r (0);
    Backprop b (r);

    // init weight vector
    std::vector< std::vector< std::vector<double> > > weights;
    weights.push_back (std::vector< std::vector<double> > ());
    weights.push_back (std::vector< std::vector<double> > ());
    weights.push_back (std::vector< std::vector<double> > ());

    weights[0].push_back (std::vector<double> ());
    weights[0].push_back (std::vector<double> ());
    weights[0].push_back (std::vector<double> ()); // bias node

    weights[1].push_back (std::vector<double> ());
    weights[1].push_back (std::vector<double> ());
    weights[1].push_back (std::vector<double> ()); // bias node

    weights[0][0].push_back (1.0); // layer 1 node 1 - layer 2 node 1
    weights[0][0].push_back (1.0); // layer 1 node 1 - layer 2 node 2
    weights[0][1].push_back (1.0); // layer 1 node 2 - layer 2 node 1
    weights[0][1].push_back (1.0); // layer 1 node 2 - layer 2 node 2
    weights[0][2].push_back (1.0); // layer 1 bias node - layer 2 node 1
    weights[0][2].push_back (1.0); // layer 1 bias node - layer 2 node 2

    weights[1][0].push_back (1.0); // layer 2 node 1 - output node
    weights[1][1].push_back (1.0); // layer 2 node 1 - output node
    weights[1][2].push_back (1.0); // layer 2 bias node - output node

    // init output vector
    std::vector< std::vector<double> > outputs;
    outputs.push_back (std::vector<double> ());
    outputs.push_back (std::vector<double> ());
    outputs.push_back (std::vector<double> ());

    outputs[0].push_back (0.0);
    outputs[0].push_back (0.0);
    outputs[0].push_back (1.0);

    outputs[1].push_back (0.0);
    outputs[1].push_back (0.0);
    outputs[1].push_back (0.0);

    outputs[2].push_back (0.0);

    b.forward(weights, outputs);
}
