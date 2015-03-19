
#include "knn.h"


void KNN::train(Matrix& features, Matrix& labels)
{
    this->k = 9;
    this->features = features;
    this->labels = labels;
}


void KNN::predict(const std::vector<double>& features, std::vector<double>& labels)
{
    if (features.size() != this->features.cols())
        ThrowError("Invalid number of attributes in given feature vector");

    std::vector<RowDistance> topFeatures;

    double maxDist = std::numeric_limits<double>::max();
    for (size_t r = 0; r < this->features.rows(); ++r)
    {
        std::vector<double> row = this->features.row(r);
        double distance = this->dist (row, features);

        if (distance < maxDist)
        {
            maxDist = replaceTop (topFeatures, r, distance);
        }
    }

    labels[0] = vote (topFeatures);
//    std::cout << "vote " << labels[0] << std::endl;
}


double KNN::dist(const std::vector<double>& feature, const std::vector<double>& input)
{
    double dist = 0.1; // prevent divide-by-zero problem
    for (size_t c = 0; c < this->features.cols(); ++c)
    {
        double target = feature[c];
        double value = input[c];
        size_t valueCount = this->features.valueCount(c);
        if (valueCount == 0) // continuous
        {
//            std::cout << "t " << target << " v " << value << " col " << c << std::endl;
            dist += pow (target - value, 2.0);
        }
        else // nominal
        {
            dist += target == value ? 0.0 : 1.0;
        }
    }

    if (dist == 0)
    {
        for (size_t i = 0; i < feature.size(); ++i)
        {
            std::cout << feature[i] << " - " << input[i] << std::endl;
        }
        std::cout << "dist " << sqrt (dist) << std::endl;
        std::cout << "------------------------" << std::endl;
    }
    return sqrt (dist);
}


double KNN::replaceTop(std::vector<RowDistance>& topFeatures, size_t newRow, double newDist)
{
    if (topFeatures.size() < this->k)
        topFeatures.push_back ( std::make_pair (newRow, newDist) );

    double max = 0.0;
    size_t maxI = 0;
    for (size_t i = 0; i < topFeatures.size(); ++i)
    {
        RowDistance tf = topFeatures[i];
        if (tf.second > max)
        {
            max = tf.second;
            maxI = i;
        }
    }
    topFeatures[maxI] = std::make_pair (newRow, newDist);

    return max;
}


bool compareVotes(const std::pair<double, double>& first, const std::pair<double, double>& second)
{
//    std::cout << "comparing " << first.first << " - " << first.second << " and " << second.first << " - " << second.second << std::endl;
    return first.second < second.second;
}


double KNN::vote(const std::vector<RowDistance>& topFeatures, bool weight)
{
    size_t valueCount = this->labels.valueCount(0);
    double denom = 0.0;

    for (size_t i = 0; i < topFeatures.size(); ++i)
    {
//        std::cout << "top dist i " << i << " - " << topFeatures[i].second << std::endl;
        if (weight)
            denom += pow (topFeatures[i].second, 2.0);
        else
            denom += 1.0;
    }

    // create map for nominal label votes
    std::map<double, size_t> valueCounts = this->labels.getValueCounts(0);
    std::map<double, double> votes;
    for (std::map<double, size_t>::iterator it = valueCounts.begin(); it != valueCounts.end(); ++it)
        votes[it->first] = 0.0;

    // this value is for continuous labels (regression)
    double estimate = 0.0;

    // loop through top feature vectors and calculate weights
    for (size_t i = 0; i < topFeatures.size(); ++i)
    {
        RowDistance tf = topFeatures[i];
        size_t row = tf.first;
        double value = this->labels.row(row)[0];
        double distanceWeight = 1.0;
        if (weight)
            distanceWeight /= pow (tf.second, 2.0);

        if (valueCount == 0) // continuous - do regression
            estimate += value * distanceWeight;
        else // nominal - classify
        {
            votes[value] += distanceWeight;
        }
    }

    if (valueCount == 0)
        return estimate / denom;
    else
    {
        for (std::map<double, double>::iterator it = votes.begin(); it != votes.end(); ++it)
            it->second /= denom;
        return (std::max_element ( votes.begin(), votes.end(), compareVotes ))->first;
    }
}





//=====================================================================
//=====================@@@@@==@===@==@@@====@===@======================
//=======================@====@===@==@==@===@@=@@======================
//=======================@=====@=@===@===@==@=@=@======================
//=======================@=====@=@===@==@===@===@======================
//=====================@@@@@====@====@@@====@===@======================
//=====================================================================
void IVDM::train(Matrix& features, Matrix& labels)
{
    trainFilter(features);
    Matrix dFeatures = discretizeFeatures(features);

    // call KNN train
    KNN::train(dFeatures, labels);
}


void IVDM::predict(const std::vector<double>& features, std::vector<double>& labels)
{
    std::vector<double> dFeatures = discretize(features);

    // call KNN predict
    KNN::predict(dFeatures, labels);
}


void IVDM::trainFilter(Matrix& features)
{
	m_bins = size_t(floor(sqrt((double)features.rows()))); // TODO play with this value

	m_featureMins.clear();
	m_featureMaxs.clear();
	size_t c = features.cols();
	m_featureMins.reserve(c);
	m_featureMaxs.reserve(c);
	for(size_t i = 0; i < c; i++)
	{
		if(features.valueCount(i) == 0)
		{
			// Compute the min and max
			m_featureMins.push_back(features.columnMin(i));
			m_featureMaxs.push_back(features.columnMax(i));
		}
		else
		{
			// Don't do nominal attributes
			m_featureMins.push_back(UNKNOWN_VALUE);
			m_featureMaxs.push_back(UNKNOWN_VALUE);
		}
	}
}


Matrix IVDM::discretizeFeatures(Matrix& features)
{
    Matrix discretized = Matrix (features);
    for (size_t i = 0; i < features.rows(); ++i)
    {
        std::vector<double> row = discretize(features.row(i));
        discretized.copyRow(row);
    }
    return discretized;
}


std::vector<double> IVDM::discretize(const std::vector<double>& before)
{
	if(before.size() != m_featureMins.size())
		ThrowError("Unexpected row size");
    std::vector<double> after;
	after.reserve(before.size());
	for(size_t c = 0; c < m_featureMins.size(); c++)
	{
		if(m_featureMins[c] == UNKNOWN_VALUE) // if the attribute is nominal...
			after.push_back(before[c]);
		else
		{
			if(before[c] == UNKNOWN_VALUE) // if the feature has an unknown value...
				after.push_back(UNKNOWN_VALUE);
			else
			{
				size_t bucket = size_t(floor((before[c] - m_featureMins[c]) * m_bins / (m_featureMaxs[c] - m_featureMins[c])));
				after.push_back(std::max((size_t)0, std::min(m_bins - 1, bucket)));
			}
		}
	}
	return after;
}

