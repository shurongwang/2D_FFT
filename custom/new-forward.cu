#include <bits/stdc++.h>
#include <cuda_fp16.h>
#include "gpu-new-forward.h"

using namespace std;

#define loop(i, to) for (int i = 0; i < to; ++i)

bool large_mask_size = 0;
__half *half_in, *half_mask;
__constant__ __half half_mask_const[4000];

__global__ void conv_forward_kernel_const_mask(float *output, const __half *input, const int B, const int M, const int C, const int H, const int W, const int K, const int S, const int H_out, const int W_out) {
	#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

	// Insert your GPU convolution kernel code here
	
	const int b = blockIdx.y, m = threadIdx.y;
	const int h = blockIdx.x, w = threadIdx.x;

	const int in_d2 = W, in_d1 = in_d2 * H, in_d0 = in_d1 * C;
	const int ms_d2 = K, ms_d1 = ms_d2 * K, ms_d0 = ms_d1 * C;
	const int d2 = W - K, d1 = in_d1 - W * K;
	
	const __half *in = input + b * in_d0 + h * S * in_d2 + w * S;
	const __half *ms = half_mask_const + m * ms_d0;

	__half res = 0;
	for (int c = 0; c < C; ++c) {
		for (int p = 0; p < K; ++p) {
			for (int q = 0; q < K; ++q) {
				res += *in * *ms; ++in; ++ms;
			}
			in += d2;
		}
		in += d1;
	}
		
	out_4d(b, m, h, w) = __half2float(res);

	#undef out_4d
}

__global__ void conv_forward_kernel(float* __restrict__ output, const __half *input, const __half *mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S, const int H_out, const int W_out) {
	#define out_4d(i3, i2, i1, i0) output[(i3) * (M * H_out * W_out) + (i2) * (H_out * W_out) + (i1) * (W_out) + i0]

	// Insert your GPU convolution kernel code here
	
	const int b = blockIdx.y, m = threadIdx.y;
	const int h = blockIdx.x, w = threadIdx.x;

	const int in_d2 = W, in_d1 = in_d2 * H, in_d0 = in_d1 * C;
	const int ms_d2 = K, ms_d1 = ms_d2 * K, ms_d0 = ms_d1 * C;
	const int d2 = W - K, d1 = in_d1 - W * K;
	
	const __half* __restrict__ in = input + b * in_d0 + h * S * in_d2 + w * S;
	const __half* __restrict__ ms = mask + m * ms_d0;

	__half res = 0;
	for (int c = 0; c < C; ++c) {
		for (int p = 0; p < K; ++p) {
			for (int q = 0; q < K; ++q) {
				res += *in * *ms; ++in; ++ms;
			}
			in += d2;
		}
		in += d1;
	}
		
	out_4d(b, m, h, w) = __half2float(res);

	#undef out_4d
}

__global__ void float_2_half(const float *f32, __half *f16, const int n) {
	int id = blockIdx.x * blockDim.x + threadIdx.x;
	if (id < n) f16[id] = __float2half(f32[id]);
}

__global__ void dbg(const __half *a, int n) {
	for (int i = 0; i < n; ++i)
		printf("%.2f\n", __half2float(a[i]));
}

__host__ void GPUInterface::conv_forward_gpu_prolog(const float *host_output, const float *host_input, const float *host_mask, float **device_output_ptr, float **device_input_ptr, float **device_mask_ptr, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
	// Allocate memory and copy over the relevant data structures to the GPU

	// We pass double pointers for you to initialize the relevant device pointers,
	//  which are passed to the other two functions.

	// Useful snippet for error checking
	// cudaError_t error = cudaGetLastError();
	// if(error != cudaSuccess)
	// {
	//     std::cout<<"CUDA error: "<<cudaGetErrorString(error)<<std::endl;
	//     exit(-1);
	// }
   
	const int H_out = (H - K) / S + 1;
	const int W_out = (W - K) / S + 1;
	const int in_size = B * C * H * W;
	const int out_size = B * M * H_out * W_out;
	const int mask_size = M * C * K * K;
	
	large_mask_size = mask_size > 1024;
	
	cudaMalloc((void**) &(*device_output_ptr), out_size * sizeof(float));
	cudaMalloc((void**) &(*device_input_ptr), in_size * sizeof(float));
	cudaMalloc((void**) &(*device_mask_ptr), mask_size * sizeof(float));
	
	cudaMalloc((void**) &half_in, in_size * sizeof(__half));
	
	__half *h_in = (__half*) malloc(in_size * sizeof(__half));
	__half *h_mask = (__half*) malloc(mask_size * sizeof(__half));

	// loop (i, in_size) h_in[i] = __float2half(host_input[i]);
	loop (i, mask_size) h_mask[i] = __float2half(host_mask[i]);
	
	// cudaMemcpy(half_in, h_in, in_size * sizeof(__half), cudaMemcpyHostToDevice);
	cudaMemcpy(*device_input_ptr, host_input, in_size * sizeof(float), cudaMemcpyHostToDevice);

	if (large_mask_size) {
		cudaMalloc((void**) &half_mask, mask_size * sizeof(__half));
		cudaMemcpy(half_mask, h_mask, mask_size * sizeof(__half), cudaMemcpyHostToDevice);
	} else
		cudaMemcpyToSymbol(half_mask_const, h_mask, mask_size * sizeof(__half));

}


__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
	// Set the kernel dimensions and call the kernel

	const int H_out = (H - K) / S + 1;
	const int W_out = (W - K) / S + 1;
	dim3 Grid(H_out, B, 1), Block(W_out, M, 1);
	
	const int in_size = B * C * H * W;
	float_2_half<<<ceil(float(in_size) / 1024), 1024>>>(device_input, half_in, in_size);
	
	cudaDeviceSynchronize();
	if (large_mask_size)
		conv_forward_kernel<<<Grid, Block>>>(device_output, half_in, half_mask, B, M, C, H, W, K, S, H_out, W_out);
	else
		conv_forward_kernel_const_mask<<<Grid, Block>>>(device_output, half_in, B, M, C, H, W, K, S, H_out, W_out);
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S)
{
	// Copy the output back to host
	const int H_out = (H - K) / S + 1;
	const int W_out = (W - K) / S + 1;
	const int out_size = B * M * H_out * W_out;

	cudaMemcpy(host_output, device_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);
   
	// Free device memory
	cudaFree(device_output);
	cudaFree(device_input);
	cudaFree(device_mask);
}


__host__ void GPUInterface::get_device_properties()
{
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	for(int dev = 0; dev < deviceCount; dev++)
	{
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, dev);

		std::cout<<"Device "<<dev<<" name: "<<deviceProp.name<<std::endl;
		std::cout<<"Computational capabilities: "<<deviceProp.major<<"."<<deviceProp.minor<<std::endl;
		std::cout<<"Max Global memory size: "<<deviceProp.totalGlobalMem<<std::endl;
		std::cout<<"Max Constant memory size: "<<deviceProp.totalConstMem<<std::endl;
		std::cout<<"Max Shared memory size per block: "<<deviceProp.sharedMemPerBlock<<std::endl;
		std::cout<<"Max threads per block: "<<deviceProp.maxThreadsPerBlock<<std::endl;
		std::cout<<"Max block dimensions: "<<deviceProp.maxThreadsDim[0]<<" x, "<<deviceProp.maxThreadsDim[1]<<" y, "<<deviceProp.maxThreadsDim[2]<<" z"<<std::endl;
		std::cout<<"Max grid dimensions: "<<deviceProp.maxGridSize[0]<<" x, "<<deviceProp.maxGridSize[1]<<" y, "<<deviceProp.maxGridSize[2]<<" z"<<std::endl;
		std::cout<<"Warp Size: "<<deviceProp.warpSize<<std::endl;
	}
}

