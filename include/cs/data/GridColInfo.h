/*
 * GridColInfo.h
 *
 *  Created on: Jul 3, 2016
 *      Author: yaison
 */

#ifndef CS_DATA_GRIDCOLINFO_H
#define CS_DATA_GRIDCOLINFO_H

#include <vector>
#include <string>

using namespace std;

namespace cs {
namespace data {

class GridColInfo {
	
private:
	
	vector<string> _diffWords;

	size_t _count;
	size_t _words;
	size_t _integers;
	size_t _floats;
	size_t _missing;
	size_t _oneCount;
	size_t _zeroCount;

	double _max;
	double _min;
	double _sum;
	double _avg;
	double _stdev;

public:
	GridColInfo();
	void fill(vector<string>& vals);

	size_t count() const;
	size_t words() const;
	const vector<string> diff_words() const;
	size_t diff_count() const;
	long int diff_idx(string& word) const;
	size_t integers() const;
	size_t floats() const;
	size_t numbers() const;
	size_t missing() const;
	size_t ones() const;
	size_t zeros() const;

	double max() const;
	double min() const;
	double sum() const;
	double avg() const;
	double stdev() const;

	void print() const;
	virtual ~GridColInfo();
};

} // namespace data
} // namespace cs

#endif // CS_DATA_GRIDCOLINFO_H
