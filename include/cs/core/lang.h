/*
 * lang.h
 *
 *  Created on: Jan 28, 2017
 *      Author: Yaison Alcantara
 */

#ifndef CS_CORE_LANG_H_
#define CS_CORE_LANG_H_

#include <stdlib.h>
#include <iostream>
#include <string>
#include <stdint.h>
#include <vector>

using namespace std;

namespace cs{
namespace core{

void check_null(const void* ptr);
void copy_float(float* src, float* dest,  size_t length);

float lower(float a, float b);
float higher(float a, float b);

void copy_float(float* src, float* dest,  size_t length, size_t start);
void print(const char* str);
void print(float val);
void print(string& str);
void println();
void println(const char* str);
void println(string& str);
void println(string str);
void println(float val);
void println(double val);
void println(long int val);
void println(int val);
void println(size_t val);


long int index_of(const vector<string>& src, string& val);
bool contains(const vector<string>& src, string& val);
long int string_to_int(string str);
float string_to_float(string str);
string trim(string const& str);



} // namespace lang
} // namespace cs
#endif // CS_CORE_LANG_H_
