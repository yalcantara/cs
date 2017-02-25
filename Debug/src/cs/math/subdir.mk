################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cs/math/CpuMatrix.cpp \
../src/cs/math/CpuVector.cpp \
../src/cs/math/GpuMatrix.cpp \
../src/cs/math/GpuVector.cpp \
../src/cs/math/Matrix.cpp \
../src/cs/math/RowView.cpp \
../src/cs/math/Vector.cpp \
../src/cs/math/math.cpp 

OBJS += \
./src/cs/math/CpuMatrix.o \
./src/cs/math/CpuVector.o \
./src/cs/math/GpuMatrix.o \
./src/cs/math/GpuVector.o \
./src/cs/math/Matrix.o \
./src/cs/math/RowView.o \
./src/cs/math/Vector.o \
./src/cs/math/math.o 

CPP_DEPS += \
./src/cs/math/CpuMatrix.d \
./src/cs/math/CpuVector.d \
./src/cs/math/GpuMatrix.d \
./src/cs/math/GpuVector.d \
./src/cs/math/Matrix.d \
./src/cs/math/RowView.d \
./src/cs/math/Vector.d \
./src/cs/math/math.d 


# Each subdirectory must supply rules for building sources it contributes
src/cs/math/%.o: ../src/cs/math/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/home/yaison/cuda-workspace/cs/include" -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src/cs/math" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/home/yaison/cuda-workspace/cs/include" -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


