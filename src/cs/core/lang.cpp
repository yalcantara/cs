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

namespace cs{
namespace core{


void check_null(const void* ptr) {
	if (ptr == nullptr || ptr == NULL) {
		throw Exception("Null pointer exception");
	}
}


void copy_float(float* src, float* dest,  size_t length){
	copy_float(src, dest, length, 0);
}


void copy_float(float* src, float* dest,  size_t length, size_t start){
	for(size_t i =0; i < length; i++){
		dest[i] = src[start + i];
	}
}

void print(const char* str){
	cout<<str;
}

void print(float val){
	cout<<val;
}

void println(){
	cout<<endl;
	fflush(stdout);
}

void println(const char* str){
	cout<<str;
	cout<<endl;
	fflush(stdout);
}

void println(string str){
	cout<<str;
	cout<<endl;
	fflush(stdout);
}

void println(float val){
	cout<<val;
	cout<<endl;
	cout.flush();
}

void println(double val){
	cout<<val;
	cout<<endl;
	cout.flush();
}

void println(long int val){
	cout<<val;
	cout<<endl;
	cout.flush();
}

void println(int val){
	cout<<val;
	cout<<endl;
	cout.flush();
}

void println(size_t val){
	cout<<val;
	cout<<endl;
	cout.flush();
}

} // namespace core
} // namespace cs
