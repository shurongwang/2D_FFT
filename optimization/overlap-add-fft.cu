#include <bits/stdc++.h>
#include <cmath>
#include <iostream>
#include "gpu-new-forward.h"

using namespace std;

#define ln '\n'
#define loop(i, to) for (int i = 0; i < to; ++i)

const float pi = 3.1415926535897932384626433832795028841971693993751058209749445923;
const int fft1d_size_log = 5;
const int fft1d_size = 32;

struct cplx {
	float real, imag;
	
	__device__ cplx() {}
	__device__ cplx(float c) { real = c, imag = 0; }
	__device__ cplx(float real, float imag) { this->real = real, this->imag = imag; }
	
	__device__ cplx operator +  (const cplx &d) const { cplx r = *this; return r += d; }
	__device__ cplx operator -  (const cplx &d) const { cplx r = *this; return r -= d; }
	__device__ cplx  operator *  (const cplx &d) const { return cplx(real * d.real - imag * d.imag, real * d.imag + imag * d.real); }
	__device__ cplx  operator /  (const cplx &d) const { cplx r = *this; return r /= d; }
	__device__ cplx& operator += (const cplx &d) { real += d.real, imag += d.imag; return *this; }
	__device__ cplx& operator -= (const cplx &d) { real -= d.real, imag -= d.imag; return *this; }
	__device__ cplx& operator *= (const cplx &d) { return *this = *this * d; }
	__device__ cplx& operator /= (const cplx &d) { return *this = *this * inverse(d); }
	
	__device__ friend ostream& operator << (ostream &out, const cplx &a) { return out << '(' << a.real << ',' << a.imag << ')'; }
	
	__device__ friend cplx inverse(const cplx &d) {
		const float t = d.real * d.real + d.imag * d.imag;
		return cplx(d.real / t, -d.imag / t);
	}
	
	__device__ inline cplx power(const cplx &w, int p) {
		if (w.real == 0 && w.imag == 0) return 0;
		cplx a = w, r = 1;
		for (; p > 0; a *= a, p >>= 1) if (p & 1) r *= a;
		return r;
	}
};

__constant__ int inv[fft1d_size];
__constant__ float rt[fft1d_size_log * fft1d_size * 2];

__device__ void swap(cplx &a, cplx &b) {
	const cplx t = a;
	a = b, b = t;
}

// [M * C], [s, s]
// expand the mask (k * k) to (s * s)
// also the mask need to be 'flipped' to do convolution
// i.e. mask(i, j) -> mask(k - 1 - i, k - 1 - j)
__global__ void expand_mask(const float *w, cplx *nw, int k, int s) {
	int r = threadIdx.x, c = threadIdx.y, id = r * s + c;
	
	const float *W = w + blockIdx.x * k * k;
	cplx *nW = nw + blockIdx.x * s * s;
	
	if (r < k && c < k) nW[id] = W[(k - 1 - r) * k + (k - 1 - c)];
	else nW[id] = 0;
}

// [M * C], [s, s]
// transpose the mask (s * s)
__device__ void transpose_mask(cplx *w, int s) {
	int r = threadIdx.x, c = threadIdx.y;
	
	cplx *W = w + blockIdx.x * s * s;
	
	if (r < c) {
		cplx *w0 = W + r * s + c, *w1 = W + c * s + r;
		swap(*w0, *w1);
	}
}

// [M * C], [s, s]
// perform a DFT to mask (s * s) in row manner
__device__ void FFT_1D_mask(cplx *a, int s) {
	int r = threadIdx.x, c = threadIdx.y, id = r * s + c;
	
	__shared__ cplx w[fft1d_size * fft1d_size];
	cplx *A = a + blockIdx.x * s * s;

	w[id] = A[id];
	__syncthreads();
	if (c > inv[c]) swap(w[id], w[r * s + inv[c]]);
	__syncthreads();
	
	cplx *w0 = w + id;	
	for (int l = 1, k = 0; l < s; l <<= 1, ++k) {
		int l2 = l << 1, i = c % l2, pos = k * s + i;
		if (i < l) {
			cplx r = cplx(rt[pos << 1], rt[pos << 1 | 1]);
			cplx *w1 = w0 + l;
			cplx tmp = *w1 * r;
			*w1 = *w0 - tmp;
			*w0 += tmp;
		}
		__syncthreads();
	}

	A[id] = w[id];
}

// [M, C], [s, s]
// perform a 2D DFT to mask (s * s)
__global__ void FFT_2D_mask(cplx *w, int s) {
	FFT_1D_mask(w, s);
	__syncthreads();
	transpose_mask(w, s);
	__syncthreads();
	FFT_1D_mask(w, s);
	__syncthreads();
	transpose_mask(w, s);
}

// [B * M, nn / s, nm / s], [s, s]
// turn each k * k grid of A into a s * s grid and forms nA
__global__ void expand(const float *a, cplx *na, int n, int m, int nn, int nm, int k, int s) {
	int R = blockIdx.y, C = blockIdx.z, r = threadIdx.x, c = threadIdx.y;
	int ni = R * s + r, nj = C * s + c, i = R * k + r, j = C * k + c;
	int nid = ni * nm + nj, id = i * m + j;

	const float *A = a + blockIdx.x * n * m;
	cplx *nA = na + blockIdx.x * nn * nm;

	if (r < k && c < k && i < n && j < m) nA[nid] = A[id];
	else nA[nid] = 0;
}

// [B * M, nn / s, nm / s], [s, s]
// transpose all s * s grids in nA
__device__ void transpose(cplx *a, int n, int m, int s) {
	int R = blockIdx.y, C = blockIdx.z, r = threadIdx.x, c = threadIdx.y;
	int i = R * s + r, j = C * s + c;
	
	cplx *A = a + blockIdx.x * n * m;

	if (r < c) {
		int ni = R * s + c, nj = C * s + r;
		cplx *w0 = &A[i * m + j], *w1 = &A[ni * m + nj];
		swap(*w0, *w1);
	}
}

// [size, nn / s, nm / s], [s, s]
// perform a DFT to all (s * s) grid in nA
__device__ void FFT_1D(cplx *a, int n, int m, int s, int type) {
	int R = blockIdx.y, C = blockIdx.z, r = threadIdx.x, c = threadIdx.y;
	int i = R * s + r, j = C * s + c;
	int id = r * s + c, idx = i * m + j;

	__shared__ cplx w[fft1d_size * fft1d_size];
	cplx *A = a + blockIdx.x * n * m;

	w[id] = A[idx];
	__syncthreads();
	if (c > inv[c]) swap(w[id], w[r * s + inv[c]]);
	__syncthreads();

	cplx *w0 = w + id;
	for (int l = 1, k = 0; l < s; l <<= 1, ++k) {
		int l2 = l << 1, i = c % l2, pos = k * s + i;
		if (i < l) {
			cplx r = cplx(rt[pos << 1], type * rt[pos << 1 | 1]);
			cplx *w1 = w0 + l, tmp = *w1 * r;
			*w1 = *w0 - tmp;
			*w0 += tmp;
		}
		__syncthreads();
	}

	A[idx] = w[id] / (type == 1 ? 1 : s);
}

// [size, nn / s, nm / s], [s, s]
// perform a 2D DFT to all (s * s) grid in nA
__global__ void FFT_2D(cplx *A, int n, int m, int s, int type) {
	FFT_1D(A, n, m, s, type);
	__syncthreads();
	transpose(A, n, m, s);
	__syncthreads();
	FFT_1D(A, n, m, s, type);
	__syncthreads();
	transpose(A, n, m, s);
}

// [B * M * C, n / s, m / s], [s, s]
// do the convolution based on the DFT result
__global__ void convolution(const cplx *a, const cplx *msk, cplx *res, int B, int C, int M, int n, int m, int s) {
	int r = threadIdx.x, c = threadIdx.y;
	int i = blockIdx.y * s + r, j = blockIdx.z * s + c, id = i * m + j;
	
	int bx = blockIdx.x, Bi = bx / (M * C), Mi = bx % (M * C) / C, Ci = bx % C, size = n * m;
	const cplx *A = a + (Bi * C + Ci) * size;
	const cplx *Msk = msk + (Mi * C + Ci) * s * s;
	cplx *Res = res + (Bi * M + Mi) * size;

	cplx &w0 = Res[id];
	const cplx w1 = A[id] * Msk[r * s + c];
	atomicAdd(&w0.real, w1.real);
	atomicAdd(&w0.imag, w1.imag);
}

// [B * M, (H - k + 1) / k, (W - k + 1) / k], [k, k]
// find the actual convolution result using the overlap-add method
__global__ void overlap_add(const cplx *a, float *res, int nn, int nm, int n, int m, int k, int s, int S) {
	int R = blockIdx.y, C = blockIdx.z, r = threadIdx.x, c = threadIdx.y;
	int ik = R * k + r, jk = C * k + c;
	
	const cplx *A = a + blockIdx.x * nn * nm;
	float *Res = res + blockIdx.x * n * m;
	
	if (ik % S == 0 && jk % S == 0) {
		int i = R * s + r + k - 1, j = C * s + c + k - 1, idx = i * nm + j;
	
		float w = A[i * nm + j].real;
		if (r) w += A[idx + (s - k) * nm].real;
		if (c) w += A[idx + s - k].real;
		if (r && c) w += A[idx + (s - k) * (nm + 1)].real;

		if (ik / S < n && jk / S < m) Res[ik / S * m + jk / S] = w;
	}
} 

int s;
cplx *nA, *nM, *nR;

__host__ void GPUInterface::conv_forward_gpu(float *device_output, const float *device_input, const float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
	int n = H, m = W, k = K;
	int nn = ((n - 1) / k + 1) * s, nm = ((m - 1) / k + 1) * s;
	int n_out = (n - k) / S + 1, m_out = (m - k) / S + 1;

	dim3 G_mask(M * C, 1, 1);
	dim3 G_in(B * C, nn / s, nm / s);
	dim3 G_conv(B * M * C, nn / s, nm / s);
	dim3 G_out(B * M, nn / s, nm / s);
	dim3 G_res(B * M, (n - k) / k + 1, (m - k) / k + 1);
	dim3 B_s(s, s, 1);
	dim3 B_k(k, k, 1);

	expand_mask<<<G_mask, B_s>>>(device_mask, nM, k, s);
	expand<<<G_in, B_s>>>(device_input, nA, n, m, nn, nm, k, s);
	
	cudaDeviceSynchronize();
	FFT_2D_mask<<<G_mask, B_s>>>(nM, s);
	FFT_2D<<<G_in, B_s>>>(nA, nn, nm, s, 1);
	
	cudaDeviceSynchronize();
	convolution<<<G_conv, B_s>>>(nA, nM, nR, B, C, M, nn, nm, s);
	
	cudaDeviceSynchronize();
	FFT_2D<<<G_out, B_s>>>(nR, nn, nm, s, -1);
	
	cudaDeviceSynchronize();
	overlap_add<<<G_res, B_k>>>(nR, device_output, nn, nm, n_out, m_out, k, s, S);
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
	
	int k = K, to = 2 * k - 1, t = 0;
	for (s = 1; s < to; s <<= 1, ++t);

	int n = H, m = W;
	int nn = ((n - 1) / k + 1) * s, nm = ((m - 1) / k + 1) * s;
	int n_out = (n - k) / S + 1, m_out = (m - k) / S + 1;

	int in_size = B * C * n * m;
	int in_size_n = B * C * nn * nm;
	int out_size = B * M * n_out * m_out;
	int out_size_n = B * M * nn * nm;
	int mask_size = M * C * k * k;
	int mask_size_n = M * C * s * s;

	cudaMalloc((void**) &(*device_output_ptr), out_size * sizeof(float));
	cudaMalloc((void**) &(*device_input_ptr), in_size * sizeof(float));
	cudaMalloc((void**) &(*device_mask_ptr), mask_size * sizeof(float));

	cudaMalloc((void**) &nA, in_size_n * sizeof(cplx));
	cudaMalloc((void**) &nM, mask_size_n * sizeof(cplx));
	cudaMalloc((void**) &nR, out_size_n * sizeof(cplx));
	
	cudaMemset(nR, 0, out_size * sizeof(cplx));
	
	int rev[fft1d_size];
	rev[0] = 0;
	loop (i, s) rev[i] = (rev[i >> 1] >> 1) | ((i & 1) << (t - 1));

	float w[fft1d_size_log * fft1d_size];
	loop (d, t) loop (i, s) {
		int l = 1 << d, pos = d * s + i;
		w[pos << 1] = cos(i * pi / l), w[pos << 1 | 1] = sin(i * pi / l);
	}
	
	cudaMemcpy(*device_input_ptr, host_input, in_size * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(*device_mask_ptr, host_mask, mask_size * sizeof(float), cudaMemcpyHostToDevice);

	cudaMemcpyToSymbol(inv, rev, s * sizeof(int));
	cudaMemcpyToSymbol(rt, w, t * s * 2 * sizeof(float));
}


__host__ void GPUInterface::conv_forward_gpu_epilog(float *host_output, float *device_output, float *device_input, float *device_mask, const int B, const int M, const int C, const int H, const int W, const int K, const int S) {
	// Copy the output back to host
	const int H_out = (H - K) / S + 1;
	const int W_out = (W - K) / S + 1;
	const int out_size = B * M * H_out * W_out;

	cudaMemcpy(host_output, device_output, out_size * sizeof(float), cudaMemcpyDeviceToHost);
	
	// Free device memory
	cudaFree(device_output);
	cudaFree(device_input);
	cudaFree(device_mask);
	
	cudaFree(nA);
	cudaFree(nM);
	cudaFree(nR);
}


__host__ void GPUInterface::get_device_properties() {
	int deviceCount;
	cudaGetDeviceCount(&deviceCount);

	for(int dev = 0; dev < deviceCount; dev++) {
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
