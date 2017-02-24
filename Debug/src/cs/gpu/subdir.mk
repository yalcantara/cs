################################################################################
# Automatically-generated file. Do not edit!
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
CPP_SRCS += \
../src/cs/gpu/gpu.cpp 

CU_SRCS += \
../src/cs/gpu/cuda_utils.cu 

CU_DEPS += \
./src/cs/gpu/cuda_utils.d 

OBJS += \
./src/cs/gpu/cuda_utils.o \
./src/cs/gpu/gpu.o 

CPP_DEPS += \
./src/cs/gpu/gpu.d 


# Each subdirectory must supply rules for building sources it contributes
src/cs/gpu/%.o: ../src/cs/gpu/%.cu
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/home/yaison/cuda-workspace/cs/include" -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src/cs/gpu" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/home/yaison/cuda-workspace/cs/include" -g -O0 -std=c++11 --compile --relocatable-device-code=false -gencode arch=compute_50,code=compute_50 -gencode arch=compute_50,code=sm_50  -x cu -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '

src/cs/gpu/%.o: ../src/cs/gpu/%.cpp
	@echo 'Building file: $<'
	@echo 'Invoking: NVCC Compiler'
	/usr/local/cuda-7.5/bin/nvcc -I"/home/yaison/cuda-workspace/cs/include" -g -O0 -std=c++11 -gencode arch=compute_50,code=sm_50  -odir "src/cs/gpu" -M -o "$(@:%.o=%.d)" "$<"
	/usr/local/cuda-7.5/bin/nvcc -I"/home/yaison/cuda-workspace/cs/include" -g -O0 -std=c++11 --compile  -x c++ -o  "$@" "$<"
	@echo 'Finished building: $<'
	@echo ' '


