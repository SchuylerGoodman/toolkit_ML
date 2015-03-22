
#include "knn.h"


void KNN::train(Matrix& features, Matrix& labels)
{
    this->k = 5;
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
}


double KNN::dist(const std::vector<double>& feature, const std::vector<double>& input)
{
    double dist = 0.1; // prevent divide-by-zero problem
    for (size_t c = 0; c < this->features.cols(); ++c)
    {
        double target = feature[c];
        double value = input[c];
        if (target == UNKNOWN_VALUE || value == UNKNOWN_VALUE)
        {
            dist += 1.0;
            continue;
        }

        size_t valueCount = this->features.valueCount(c);
        if (valueCount == 0) // continuous
        {
            dist += pow (target - value, 2.0);
        }
        else // nominal
        {
            dist += target == value ? 0.0 : 1.0;
        }
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
    return first.second < second.second;
}


double KNN::vote(const std::vector<RowDistance>& topFeatures, bool weight)
{
    size_t valueCount = this->labels.valueCount(0);
    double denom = 0.0;

    for (size_t i = 0; i < topFeatures.size(); ++i)
    {
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
    {
        return estimate / denom;
    }
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
    this->labelValueCounts = labels.getValueCounts(0);

    features.useUnknown();

    trainFilter (features);
    Matrix dFeatures = discretizeFeatures (features);

    for (size_t a = 0; a < dFeatures.cols(); ++a)
    {
        std::map<double, size_t> valueCounts = dFeatures.getValueCounts (a);
        std::map<double, std::map<double, double> > probs_a;

        for (std::map<double, size_t>::iterator it = valueCounts.begin(); it != valueCounts.end(); ++it)
        {
            std::map<double, double> probs_av;
            probs_a[it->first] = probs_av;
        }

        for (size_t i = 0; i < dFeatures.rows(); ++i)
        {
            double value = dFeatures.row(i)[a];
            if (probs_a.find(value) == probs_a.end())
                ThrowError ("Something is wrong with getValueCounts");

            double c = labels.row(i)[0];
            if (probs_a[value].find(c) == probs_a[value].end())
                probs_a[value][c] = 0.0;

            probs_a[value][c] += 1.0;
        }

        for (std::map<double, std::map<double, double> >::iterator it = probs_a.begin(); it != probs_a.end(); ++it)
        {
            double value = it->first;
            std::map<double, double> probs_av = it->second;
            for (std::map<double, double>::iterator dit = probs_av.begin(); dit != probs_av.end(); ++dit)
                dit->second = dit->second / valueCounts[value]; // p(a,v,c) = N(a,v,c) / N(a,v)
        }
        this->probabilities[a] = probs_a;
    }

    // call KNN train
    KNN::train (features, labels);
}


void IVDM::predict(const std::vector<double>& features, std::vector<double>& labels)
{
    std::vector<double> dFeatures = discretize (features);

    // call KNN predict
    KNN::predict (features, labels);
}


double IVDM::dist(const std::vector<double>& feature, const std::vector<double>& input) {

    double distance = 0.1;
    double Pauc, Pau1c, Mau, Mau1;
    for (size_t a = 0; a < this->features.cols(); ++a)
    {
        double target = feature[a];
        double value = input[a];
        double min = m_featureMins[a];
        double width = m_featureWidths[a];

        if (target == UNKNOWN_VALUE)
            target = m_bins;
        if (value == UNKNOWN_VALUE)
            value = m_bins;

        std::map<double, std::map<double, double> >& probs_a = probabilities[a];

        size_t valueCount = this->features.valueCount(a);
        for (std::map<double, size_t>::iterator it = labelValueCounts.begin(); it != labelValueCounts.end(); ++it)
        {
            double label = it->first;
            if (valueCount == 0) // continuous
            {
                size_t targetBin = getBin(target, min, width);
                Pauc = probs_a[targetBin][label];
                Pau1c = probs_a[targetBin + 1][label];
                Mau = min + width * (targetBin + 0.5);
                Mau1 = min + width * (targetBin + 1.5);
                double Pacx = Pauc + ( (target - Mau) / (Mau1 - Mau) ) * (Pau1c - Pauc);
//                double Pacx = calcPac(target, targetBin, a, label);

                size_t valueBin = getBin(value, min, width);
                Pauc = probs_a[valueBin][label];
                Pau1c = probs_a[valueBin + 1][label];
                Mau = min + width * (targetBin + 0.5);
                Mau1 = min + width * (targetBin + 1.5);
                double Pacy = Pauc + ( (value - Mau) / (Mau1 - Mau) ) * (Pau1c - Pauc);
//                double Pacy = calcPac(value, valueBin, a, label);

                distance += pow (Pacx - Pacy , 2);
            }
            else // nominal
            {
                distance += pow (target - value, 2);
            }
        }
    }
    return distance;
}


void IVDM::trainFilter(Matrix& features)
{
	m_bins = (size_t) floor ( sqrt ( (double) features.rows() ) ); // TODO play with this value

	m_featureMins.clear();
	m_featureMaxs.clear();
	size_t c = features.cols();
	m_featureMins.reserve(c);
	m_featureMaxs.reserve(c);
	for (size_t i = 0; i < c; i++)
	{
		if (features.valueCount(i) == 0)
		{
			// Compute the min and max
			m_featureMins.push_back (features.columnMin(i));
			m_featureMaxs.push_back (features.columnMax(i));
            m_featureWidths.push_back (std::abs (features.columnMax(i) - features.columnMin(i)) / m_bins);
		}
		else
		{
			// Don't do nominal attributes
			m_featureMins.push_back (UNKNOWN_VALUE);
			m_featureMaxs.push_back (UNKNOWN_VALUE);
            m_featureWidths.push_back (UNKNOWN_VALUE);
		}
	}
}


Matrix IVDM::discretizeFeatures(Matrix& features)
{
    Matrix discretized = Matrix (features);
    for (size_t i = 0; i < features.rows(); ++i)
    {
        std::vector<double> row = discretize (features.row(i));
        discretized.copyRow (row);
    }
    return discretized;
}


std::vector<double> IVDM::discretize(const std::vector<double>& before)
{
	if(before.size() != m_featureMins.size())
		ThrowError ("Unexpected row size");
    std::vector<double> after;
	after.reserve (before.size());
	for(size_t c = 0; c < m_featureMins.size(); c++)
	{
        double width = m_featureWidths[c];
		if(m_featureMins[c] == UNKNOWN_VALUE) // if the attribute is nominal...
			after.push_back (before[c]);
		else
		{
			if(before[c] == UNKNOWN_VALUE) // if the feature has an unknown value...
				after.push_back ( m_bins );
			else
			{
                size_t bucket = getBin(before[c], m_featureMins[c], width);
				after.push_back ( std::max ( (size_t)0, std::min (m_bins - 1, bucket) ) );
			}
		}
	}
	return after;
}


size_t IVDM::getBin(double x, double min, double width)
{
    if (x < min)
        return 0;
    return size_t( floor ( (x - min) / width ) );
}


double IVDM::calcPac(double x, size_t u, size_t a, double c)
{
    double Pauc = this->probabilities[a][u][c];

    double min = m_featureMins[a];
    double width = m_featureWidths[a];

    double Mau = min + width * (u + 0.5);
    double Mau1 = min + width * (u + 1.5);

    double Pau1c = this->probabilities[a][u + 1][c];

    return Pauc + ( (x - Mau) / (Mau1 - Mau) ) * (Pau1c - Pauc);
}


double IVDM::calcMid(size_t a, size_t u)
{
    return m_featureMins[a] + m_featureWidths[a] * (u + 0.5);
}

