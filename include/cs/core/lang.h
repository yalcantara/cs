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

using namespace std;

namespace cs{
namespace core{

void check_null(const void* ptr);
void copy_float(float* src, float* dest,  size_t length);

void copy_float(float* src, float* dest,  size_t length, size_t start);
void print(const char* str);
void print(float val);
void println();
void println(const char* str);
void println(string str);
void println(float val);
void println(double val);
void println(long int val);
void println(int val);
void println(size_t val);

} // namespace lang
} // namespace cs
#endif // CS_CORE_LANG_H_
