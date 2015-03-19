// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "matrix.h"
#include "rand.h"
#include "error.h"
#include <fstream>

using std::string;
using std::ifstream;
using std::map;
using std::vector;

Matrix::Matrix(Matrix& that)
{
	m_attr_name = that.m_attr_name;
	m_str_to_enum = that.m_str_to_enum;
	m_enum_to_str = that.m_enum_to_str;
}

Matrix::Matrix(const Matrix& that)
{
    m_attr_name = that.m_attr_name;
	m_str_to_enum = that.m_str_to_enum;
	m_enum_to_str = that.m_enum_to_str;
}

void Matrix::setSize(size_t rows, size_t cols)
{
	// Make space for the data
	m_data.resize(rows);
	for(size_t i = 0; i < rows; i++)
		m_data[i].resize(cols);

	// Set the meta-data
	m_filename = "";
	m_attr_name.resize(cols);
	m_str_to_enum.resize(cols);
	m_enum_to_str.resize(cols);
	for(size_t i = 0; i < cols; i++)
	{
		m_str_to_enum[i].clear();
		m_enum_to_str[i].clear();
	}
}

std::vector<size_t> Matrix::allAttrValues(size_t attr)
{
    std::map<std::string, size_t> attrMap = m_str_to_enum[attr];
    std::vector<size_t> attrValues;

    for (map<std::string, size_t>::iterator it = attrMap.begin(); it != attrMap.end(); ++it)
    {
        attrValues.push_back(it->second);
    }

    return attrValues;
}

void Matrix::shuffleRows(Rand& r, Matrix* pBuddy)
{
	if(pBuddy)
	{
		for(size_t n = rows(); n > 0; n--)
		{
			size_t i = (size_t)r.next(n);
			std::swap(m_data[i], m_data[n - 1]);
			std::swap((*pBuddy)[i], (*pBuddy)[n - 1]);
		}
	}
	else
	{
		for(size_t n = rows(); n > 0; n--)
			std::swap(m_data[(size_t)r.next(n)], m_data[n - 1]);
	}
}

double Matrix::columnMean(size_t col)
{
	double sum = 0.0;
	size_t count = 0;
	std::vector< std::vector<double> >::iterator it;
	for(it = m_data.begin(); it != m_data.end(); it++)
	{
		double val = (*it)[col];
		if(val != UNKNOWN_VALUE)
		{
			sum += val;
			count++;
		}
	}
	return sum / count;
}

double Matrix::columnMin(size_t col)
{
	double m = 1e300;
	std::vector< std::vector<double> >::iterator it;
	for(it = m_data.begin(); it != m_data.end(); it++)
	{
		double val = (*it)[col];
		if(val != UNKNOWN_VALUE)
			m = std::min(m, val);
	}
	return m;
}

double Matrix::columnMax(size_t col)
{
	double m = -1e300;
	std::vector< std::vector<double> >::iterator it;
	for(it = m_data.begin(); it != m_data.end(); it++)
	{
		double val = (*it)[col];
		if(val != UNKNOWN_VALUE)
			m = std::max(m, val);
	}
	return m;
}

double Matrix::mostCommonValue(size_t col)
{
	map<double, size_t> counts = getValueCounts(col);

	size_t valueCount = 0;
	double value = 0;
	for(map<double, size_t>::iterator i = counts.begin(); i != counts.end(); i++)
	{
		if(i->first != UNKNOWN_VALUE && i->second > valueCount)
		{
			value = i->first;
			valueCount = i->second;
		}
	}
	return value;
}

map<double, size_t> Matrix::getValueCounts(size_t col)
{
    map<double, size_t> counts;
	vector< vector<double> >::iterator it;
	for(it = m_data.begin(); it != m_data.end(); it++)
	{
		double val = (*it)[col];
        map<double, size_t>::iterator pair = counts.find(val);
        if(pair == counts.end())
            counts[val] = 1;
        else
            pair->second++;
    }

    return counts;
}

void Matrix::copyRow(vector<double>& row)
{
	if(row.size() != cols())
		ThrowError("mismatching size");
	m_data.push_back(row);
}

void Matrix::copyPart(Matrix& that, size_t rowBegin, size_t colBegin, size_t rowCount, size_t colCount)
{
	if(rowBegin + rowCount > that.rows() || colBegin + colCount > that.cols())
		ThrowError("out of range");

	// Copy the specified region of meta-data
	if(cols() != colCount)
		setSize(0, colCount);
	for(size_t i = 0; i < colCount; i++)
	{
		m_attr_name[i] = that.m_attr_name[colBegin + i];
		m_str_to_enum[i] = that.m_str_to_enum[colBegin + i];
		m_enum_to_str[i] = that.m_enum_to_str[colBegin + i];
	}

	// Copy the specified region of data
	size_t rowsBefore = m_data.size();
	m_data.resize(rowsBefore + rowCount);
	for(size_t i = 0; i < rowCount; i++)
	{
		vector<double>::iterator itIn = that[rowBegin + i].begin() + colBegin;
		m_data[rowsBefore + i].resize(colCount);
		vector<double>::iterator itOut = m_data[rowsBefore + i].begin();
		for(size_t j = 0; j < colCount; j++)
			*itOut++ = *itIn++;
	}
}

string toLower(string strToConvert)
{
	//change each element of the string to lower case
	for(size_t i = 0; i < strToConvert.length(); i++)
		strToConvert[i] = tolower(strToConvert[i]);
	return strToConvert;//return the converted string
}

void Matrix::loadARFF(string fileName)
{
	size_t lineNum = 0;
	string line;                 //line of input from the arff file
	ifstream inputFile;          //input stream
	map <string, size_t> tempMap;   //temp map for int->string map (attrInts)
	map <size_t, string> tempMapS;  //temp map for string->int map (attrString)
	size_t attrCount = 0;           //Count number of attributes

	inputFile.open ( fileName.c_str() );

	//Ensure that the file name is correct
	if ( !inputFile )
		ThrowError ( "failed to open the file: ", fileName );
	//Parse the file. save data in data variable
	while ( !inputFile.eof() && inputFile )
	{
		//Iterate through each line of the file
		getline ( inputFile, line );
		lineNum++;
		if ( toLower ( line ).find ( "@relation" ) == 0 )
		{
			m_filename = line.substr ( line.find_first_of ( " " ) );
		}
		else if ( toLower ( line ).find ( "@attribute" ) == 0 )
		{
			line = line.substr ( line.find_first_of ( " \t" ) + 1 );

			//Parse the attribute name
			string attrName = line.substr ( 0, line.find_first_of ( " \t" ) );
			m_attr_name.push_back ( attrName );

			line = line.substr ( attrName.size() );
			string value = line.substr ( line.find_first_not_of ( " \t" ) );
			tempMap.clear();
			tempMapS.clear();

			//If the attribute is nominal
			if ( value.find_first_of ( "{" ) == 0 )
			{
				int firstComma;
				int firstSpace;
				int firstLetter;
				value = value.substr ( 1, value.find_last_of ( "}" ) - 1 );
				size_t valCount = 0;
				string tempValue;

				//Parse the attributes--push onto the maps
				while ( ( firstComma = value.find_first_of ( "," ) ) > -1 )
				{
					firstLetter = value.find_first_not_of ( " \t," );

					value = value.substr ( firstLetter );
					firstComma = value.find_first_of ( "," );
					firstSpace = value.find_first_of ( " \t" );
					tempMapS[valCount] = value.substr ( 0, firstComma );
					string valName = value.substr ( 0, firstComma );
					valName = valName.substr ( 0, valName.find_last_not_of(" \t") + 1);
					tempMap[valName] = valCount++;
					firstComma = ( firstComma < firstSpace &&
						firstSpace < ( firstComma + 2 ) ) ? firstSpace :
						firstComma;
					value = value.substr ( firstComma + 1 );
				}

				//Push final attribute onto the maps
				firstLetter = value.find_first_not_of ( " \t," );
				value = value.substr ( firstLetter );
				string valName = value.substr ( 0, value.find_last_not_of(" \t") + 1);
				tempMapS[valCount] = valName;
				tempMap[valName] = valCount++;
				m_str_to_enum.push_back ( tempMap );
				m_enum_to_str.push_back ( tempMapS );
			}
			else
			{
				//The attribute is continuous
				m_str_to_enum.push_back ( tempMap );
				m_enum_to_str.push_back ( tempMapS );
			}
			attrCount++;
		}
		else if ( toLower ( line ).find ( "@data" ) == 0 )
		{
			vector<double> temp; //Holds each line of data
			temp.reserve(attrCount);

			//Clear the data
			m_data.clear();

			//Read through the rest of the file
			while ( !inputFile.eof() )
			{
				getline ( inputFile, line );
				lineNum++;
				line = line.substr ( 0, line.find_first_of ( "\r\n" ) );
				//Check for commented out lines
				if ( line.find ( "%" ) != 0 && line != "" )
				{
					//Push each data item onto the back of the temp vector
					for ( size_t i = 0; i < attrCount; i++ )
					{
						size_t vals = valueCount ( i );
						string val = line.substr ( line.find_first_not_of ( " \t" ), line.find_first_of ( ",\n\r" ) );
						if ( vals > 0 ) //if the attribute is nominal...
						{
							if ( val == "?" )
							{
								temp.push_back(UNKNOWN_VALUE);
							}
							else
							{
								map<string, size_t>::iterator it = m_str_to_enum[i].find ( val );
								if ( it == m_str_to_enum[i].end() && m_str_to_enum[i].size() > 1 )
								{
									m_str_to_enum[i][val] = ( m_enum_to_str[i].end()->first )-1;
									m_enum_to_str[i][ ( m_enum_to_str[i].end()->first )-1] = val;
								}
								temp.push_back ( m_str_to_enum[i][val] );
							}
						}
						else
						{
							//the attribute is continuous
							std::stringstream convert ( val );
							double tempDouble;
							if ( val == "?" )
							{
								tempDouble = UNKNOWN_VALUE;
							}
							else
							{
								convert >> tempDouble;
							}
							temp.push_back ( tempDouble );
						}
						if ( i != attrCount - 1 )
						{
							size_t pos = line.find_first_of ( ", \t" );
							if(pos == string::npos)
								ThrowError("Expected more elements on line ", to_str(lineNum));
							line = line.substr ( pos );
							pos = line.find_first_not_of ( ", \t" );
							if(pos == string::npos)
								ThrowError("Expected something after the last delimiter on line ", to_str(lineNum));
							line = line.substr ( pos );
						}
					}
					if(temp.size() != cols())
						ThrowError("Expected ", to_str(cols()), " elements on line ", to_str(lineNum), ". Found ", to_str(lineNum), ".");
					m_data.push_back ( temp );
					temp.clear();
				}
			}
		}
	}
}

void Matrix::useUnknown()
{
    size_t c = cols();
    for (size_t i = 0; i < c; ++i)
    {
        vector<size_t> rowIs = getRowsWithAttrValues(i, UNKNOWN_VALUE);
        if (rowIs.size() > 0)
        {
            size_t vals = valueCount ( i );
            if ( vals > 0 ) //if the attribute is nominal... 
            {
                std::string val = "?";
                size_t e = m_enum_to_str[i].end()->first;
                map<string, size_t>::iterator it = m_str_to_enum[i].find ( val );
                
                // add unknown as enum
                if ( it == m_str_to_enum[i].end() && m_str_to_enum[i].size() > 1 )
                {
                    m_str_to_enum[i][val] = e;
                    m_enum_to_str[i][e] = val;
                }

                // convert data values to enum instead
                for (size_t j = 0; j < rowIs.size(); ++j)
                {
                    size_t row = rowIs[j];
                    if (this->row(row)[i] == UNKNOWN_VALUE)
                        this->row(row)[i] = e;
                }
            }
            //continuous values are not enumerated, so it is ok for them to have UNKNOWN_VALUE
        }
    }
}

void Matrix::setAll(double val)
{
	size_t c = cols();
	std::vector< std::vector<double> >::iterator it;
	for(it = m_data.begin(); it != m_data.end(); it++)
		it->assign(c, val);
}

void Matrix::checkCompatibility(Matrix& that)
{
	size_t c = cols();
	if(that.cols() != c)
		ThrowError("Matrices have different number of columns");
	for(size_t i = 0; i < c; i++)
	{
		if(valueCount(i) != that.valueCount(i))
			ThrowError("Column ", to_str(i), " has mis-matching number of values");
	}
}

void Matrix::makeContinuousAttrsNominal(size_t values)
{
	size_t c = cols();
	for(size_t i = 0; i < c; i++)
	{
		if(valueCount(i) == 0)
		{
			for(size_t j = 0; j < values; j++)
				m_enum_to_str[i][j] = "";
		}
	}
}

std::vector<size_t> Matrix::getRowsWithAttrValues(size_t attr, double value)
{
    size_t r = rows();
    if (attr >= cols())
        ThrowError("getRowsWithAttrValues::Invalid attr number");

    std::vector<size_t> result;
    for (size_t i = 0; i < r; ++i)
    {
        if (row(i)[attr] == value)
        {
            result.push_back(i);
        }
    }
    return result;
}

std::vector<size_t> Matrix::copyReduce(Matrix& reduced, size_t attr, double value)
{
    reduced.setSize(0, this->cols());
    reduced = Matrix(*this);

    // remove attribute and value info not in new matrix
    std::vector<std::string>::iterator attr_name_it = reduced.m_attr_name.begin() + attr;
    reduced.m_attr_name.erase(attr_name_it);

    std::vector< std::map<std::string, size_t> >::iterator str_to_enum_it = reduced.m_str_to_enum.begin() + attr;
    reduced.m_str_to_enum.erase(str_to_enum_it);

    std::vector< std::map<size_t, std::string> >::iterator enum_to_str_it = reduced.m_enum_to_str.begin() + attr;
    reduced.m_enum_to_str.erase(enum_to_str_it);

    // copy the rows we want
    std::vector<size_t> keepRows = getRowsWithAttrValues(attr, value);
    for (size_t i = 0; i < keepRows.size(); ++i)
    {
        // copy this row
        std::vector<double> row = std::vector<double> (this->row(keepRows[i]));

        // remove attr column from every row
        std::vector<double>::iterator row_it = row.begin() + attr;
        row.erase(row_it, row_it + 1);

        // copy row (again?) into reduced matrix
        reduced.copyRow(row);
    }
    //std::cout << "copyReduce - " << this->rows() << " x " << this->cols() << " --> " << reduced.rows() << " x " << reduced.cols() << std::endl;

    return keepRows;
}

void Matrix::copyReduce(Matrix& reduced, std::vector<size_t> rows)
{
    reduced.setSize(0, this->cols());
    reduced = Matrix(*this);

    for (size_t i = 0; i < rows.size(); ++i)
    {
        // copy this row
        std::vector<double> row = std::vector<double> (this->row(rows[i]));

        // copy row (again?) into reduced matrix
        reduced.copyRow(row);
    }
}

void Matrix::toCSV()
{
    size_t rows = this->rows();
    size_t cols = this->cols();
//    std::cout << rows << " _____ " << cols << std::endl;
    for (size_t i = 0; i < rows; ++i)
    {
        if (cols > 0)
        {
            std::cout << this->row(i)[0];
            for (size_t j = 1; j < cols; ++j)
            {
                std::cout << "," << this->row(i)[j];
            }
            std::cout << std::endl;
        }
    }
}
