/*
 * GridInfo.h
 *
 *  Created on: Jul 3, 2016
 *      Author: yaison
 */

#ifndef CS_DATA_GRIDINFO_H_
#define CS_DATA_GRIDINFO_H_

#include <stdlib.h>
#include <vector>
#include <string>

#include <cs/data/GridColInfo.h>

using namespace std;

namespace cs {
namespace data{

class GridInfo {
private:
	GridColInfo* colInfo;
	const size_t _cols;

	vector<string> get_column_values(vector<string> data, size_t cols, size_t col);

public:
	GridInfo(size_t cols);

	void fill(vector<string>& data);
	void fill(vector<string>& data, size_t col);

	size_t rows() const;
	size_t cols() const;
	size_t diff_count(size_t col) const;
	const vector<string> diff_words(size_t col) const;
	long int diff_idx(size_t col, string& word) const;

	bool has_missing(size_t col) const;
	bool is_word(size_t col) const;
	bool is_integer(size_t col) const;
	bool is_numeric(size_t col) const;
	bool is_float(size_t col) const;
	bool is_boolean(size_t col) const;

	double sum(size_t col) const;
	double max(size_t col) const;
	double min(size_t col) const;
	double avg(size_t col) const;
	double stdev(size_t col) const;

	virtual ~GridInfo();
};

} // namespace data
} // namespace cs

#endif // CS_DATA_GRIDINFO_H_
