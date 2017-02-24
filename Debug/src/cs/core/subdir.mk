################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cs/core/Exception.cpp \
../src/cs/core/lang.cpp 

OBJS += \
./src/cs/core/Exception.o \
./src/cs/core/lang.o 

CPP_DEPS += \
./src/cs/core/Exception.d \
./src/cs/core/lang.d 


# Each subdirectory must supply rules for building sources it contributes
src/cs/core/%.o: ../src/cs/core/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/home/yaison/cuda-workspace/cs/include" -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src/cs/core" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/home/yaison/cuda-workspace/cs/include" -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


