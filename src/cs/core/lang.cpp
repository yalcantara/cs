/*
 * lang.cpp
 *
 *  Created on: Jan 30, 2017
 *      Author: Yaison Alcantara
 */

#include <cs/core/Exception.h>
#include <stddef.h>
#include <cs/core/lang.h>
#include <cstdio>
#include <iostream>
#include <string>
#include <algorithm>

namespace cs {
namespace core {

void check_null(const void* ptr) {
	if (ptr == nullptr || ptr == NULL) {
		throw Exception("Null pointer exception");
	}
}

/*
 * Returns the value of a if a > b, and b if b > a. Note that for this method
 * any value is higher than NAN.
 */
float higher(float a, float b) {
	if (isnan(a)) {
		if (isnan(b)) {
			return NAN;
		}
		return b;
	}
	
	if (isnan(b)) {
		return a;
	}
	
	if (a > b) {
		return a;
	}
	
	return b;
}

/*
 * Returns the value of a if a < b, and b if b < a. Note that for this method
 * any value is lower than NAN.
 */
float lower(float a, float b) {
	if (isnan(a)) {
		if (isnan(b)) {
			return NAN;
		}
		
		return b;
	}
	
	if (isnan(b)) {
		return a;
	}
	
	if (a < b) {
		return a;
	}
	
	return b;
}

void copy_float(float* src, float* dest, size_t length) {
	copy_float(src, dest, length, 0);
}

void copy_float(float* src, float* dest, size_t length, size_t start) {
	for (size_t i = 0; i < length; i++) {
		dest[i] = src[start + i];
	}
}

void print(const char* str) {
	cout << str;
}

void print(float val) {
	cout << val;
}

void print(string& str) {
	cout << str;
}



void print(vector<string>& vec) {
	
	print("[");
	for (size_t i = 0; i < vec.size(); i++) {
		string& e = vec[i];
		print(e);
		if (i < vec.size()) {
			print(", ");
		}
	}
	print("]");
}

void println() {
	cout << endl;
	fflush(stdout);
}

void println(const char* str) {
	cout << str;
	cout << endl;
	fflush(stdout);
}

void println(string& str) {
	cout << str;
	cout << endl;
	fflush(stdout);
}

void println(string str) {
	cout << str;
	cout << endl;
	fflush(stdout);
}

void println(float val) {
	cout << val;
	cout << endl;
	cout.flush();
}

void println(double val) {
	cout << val;
	cout << endl;
	cout.flush();
}

void println(long int val) {
	cout << val;
	cout << endl;
	cout.flush();
}

void println(int val) {
	cout << val;
	cout << endl;
	cout.flush();
}

void println(size_t val) {
	cout << val;
	cout << endl;
	cout.flush();
}



//We need long int cuz it can allow negative values (-1) when
//the item could not be found.
long int index_of(const vector<string>& src, string& val) {
	auto r = find(src.begin(), src.end(), val);
	if (r == src.end()) {
		return -1;
	}
	
	long int pos = distance(src.begin(), r);
	return pos;
}

bool contains(const vector<string>& src, string& val) {
	return index_of(src, val) >= 0;
}

long int string_to_int(string str) {
	return atol(str.c_str());
}

float string_to_float(string str) {
	return atof(str.c_str());
}

string trim(string const& str) {
	if (str.empty())
		return str;
	
	size_t firstScan = str.find_first_not_of(' ');
	size_t first = firstScan == string::npos ? str.length() : firstScan;
	size_t last = str.find_last_not_of(' ');
	return str.substr(first, last - first + 1);
}


} // namespace core
} // namespace cs
