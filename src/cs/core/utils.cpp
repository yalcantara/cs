/**
 * Copyright 1993-2012 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 */
#include <stdio.h>
#include <stdlib.h>
#include <algorithm>
#include <string>

using namespace std;

namespace cs {
namespace core {

//IO
const string ffull(const char* path) {
	
	FILE* f = fopen(path, "r");
	
	if (f == NULL) {
		fprintf(stderr, "file not found at: %s\n", path);
		fflush(stderr);
		return NULL;
	}
	
	char c;
	
	//A trick to determine how many characters a file has
	size_t count = 0;
	while ((c = fgetc(f))) {
		if (c == EOF) {
			break;
		}
		count++;
	}
	
	char* content = (char*) malloc(sizeof(char) * (count + 1));
	content[count] = 0;
	
	rewind(f);
	size_t r = fread(content, sizeof(char), count, f);
	
	fclose(f);
	
	const string s(content);
	free(content);
	
	
	return s;	
}

} //namespace core
} //namespace cs
