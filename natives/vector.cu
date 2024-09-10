#include <stdio.h>
#include <cuda_runtime.h>
#include <jni.h>

#define CHECK_CUDA_ERROR(call)                                                           \
    {                                                                                    \
        const cudaError_t error = call;                                                  \
        if (error != cudaSuccess)                                                        \
        {                                                                                \
            fprintf(stderr, "CUDA Error: %s:%d, ", __FILE__, __LINE__);                  \
            fprintf(stderr, "code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
            return;                                                                      \
        }                                                                                \
    }

#define BLOCK_SIZE 256

/**
 *
 *  Kernel functions for vector operations
 *
 */

// Negate Kernel
__global__ void negateKernel(float *d_input, float *d_output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        d_output[idx] = -d_input[idx];
    }
}

// Add Kernel
__global__ void addKernel(float *d_A, float *d_B, float *d_C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
}

// Sub Kernel
__global__ void subKernel(float *d_A, float *d_B, float *d_C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        d_C[idx] = d_A[idx] - d_B[idx];
    }
}

// Mult Scalar Kernel
__global__ void multScalarKernel(float *d_input, float factor, float *d_output, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        d_output[idx] = d_input[idx] * factor;
    }
}

// Mult Vector Kernel
__global__ void multVectorKernel(float *d_A, float *d_B, float *d_C, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size)
    {
        d_C[idx] = d_A[idx] * d_B[idx];
    }
}

// Matrix Multiplication Kernel
__global__ void matMulKernel(float *d_A, float *d_B, float *d_C, int A_rows, int A_cols, int B_cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < A_rows && col < B_cols)
    {
        float value = 0;
        for (int k = 0; k < A_cols; ++k)
        {
            value += d_A[row * A_cols + k] * d_B[k * B_cols + col];
        }
        d_C[row * B_cols + col] = value;
    }
}

// Dot Product Kernel with Reduction
__global__ void dotKernel(float *d_A, float *d_B, float *d_result, int size)
{
    __shared__ float temp[BLOCK_SIZE];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int threadId = threadIdx.x;

    float sum = 0;
    if (idx < size)
    {
        sum = d_A[idx] * d_B[idx];
    }

    temp[threadId] = sum;
    __syncthreads();

    // Reduction to sum all the elements
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) // this only
    {
        if (threadId < stride)
        {
            temp[threadId] += temp[threadId + stride];
        }
        __syncthreads();
    }

    if (threadId == 0)
    {
        atomicAdd(d_result, temp[0]);
    }
}

/**
 *
 *  Helper functions for common tasks
 *
 */

// Error checking helper
void checkCudaError(cudaError_t error, const char *msg)
{
    if (error != cudaSuccess)
    {
        printf("CUDA Error: %s - %s\n", msg, cudaGetErrorString(error));
        cudaDeviceReset();
    }
}

// Allocate memory and copy data from host to device
float *allocateJfloatArrayAndCopyToDevice(JNIEnv *env, jfloatArray jData, int size)
{
    jfloat *h_data = env->GetFloatArrayElements(jData, 0);
    float *d_data;
    checkCudaError(cudaMalloc(&d_data, size * sizeof(float)), "Failed to allocate GPU memory");
    checkCudaError(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy data to GPU");
    env->ReleaseFloatArrayElements(jData, h_data, 0);
    return d_data;
}

// Allocate memory and copy data from host to device
float *allocateJobjectArrayAndCopyToDevice(JNIEnv *env, jobjectArray jData, int rows, int cols)
{
    int size = rows * cols;
    float *h_data = new float[size];

    for (int i = 0; i < rows; ++i)
    {
        jfloatArray rowArray = (jfloatArray)env->GetObjectArrayElement(jData, i);
        jfloat *rowData = env->GetFloatArrayElements(rowArray, 0);
        for (int j = 0; j < cols; ++j)
        {
            h_data[i * cols + j] = rowData[j];
        }
        env->ReleaseFloatArrayElements(rowArray, rowData, 0);
        env->DeleteLocalRef(rowArray);
    }

    float *d_data;
    checkCudaError(cudaMalloc(&d_data, size * sizeof(float)), "Failed to allocate GPU memory");
    checkCudaError(cudaMemcpy(d_data, h_data, size * sizeof(float), cudaMemcpyHostToDevice), "Failed to copy data to GPU");

    delete[] h_data;
    return d_data;
}

// Copy data from device to host and return it as a new float array
jfloatArray copyDeviceToHostAndCreateArray(JNIEnv *env, float *d_data, int size)
{
    jfloatArray resultArray = env->NewFloatArray(size);
    jfloat *h_output = new jfloat[size];
    checkCudaError(cudaMemcpy(h_output, d_data, size * sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy result to host");
    env->SetFloatArrayRegion(resultArray, 0, size, h_output);
    delete[] h_output;
    return resultArray;
}

// Utility to allocate device memory and check for errors
float *allocateDeviceMemory(JNIEnv *env, jfloatArray array, int size)
{
    jfloat *hostArray = env->GetFloatArrayElements(array, NULL);
    if (hostArray == NULL)
    {
        fprintf(stderr, "Error: Failed to get array elements from Java.\n");
        return nullptr;
    }

    float *deviceArray;
    const cudaError_t mallocError = cudaMalloc((void **)&deviceArray, size * sizeof(float));
    if (mallocError != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, mallocError, cudaGetErrorString(mallocError));
        return nullptr;
    }

    const cudaError_t memcpyError = cudaMemcpy(deviceArray, hostArray, size * sizeof(float), cudaMemcpyHostToDevice);
    if (memcpyError != cudaSuccess)
    {
        fprintf(stderr, "CUDA Error: %s:%d, code: %d, reason: %s\n", __FILE__, __LINE__, memcpyError, cudaGetErrorString(memcpyError));
        cudaFree(deviceArray); // Clean up memory in case of failure
        return nullptr;
    }

    env->ReleaseFloatArrayElements(array, hostArray, 0);
    return deviceArray; // Return the allocated and populated device array
}

// Utility to copy memory back to host and release device memory
void releaseDeviceMemory(JNIEnv *env, jfloatArray result, float *deviceArray, int size)
{
    jfloat *hostResult = env->GetFloatArrayElements(result, NULL);
    CHECK_CUDA_ERROR(cudaMemcpy(hostResult, deviceArray, size * sizeof(float), cudaMemcpyDeviceToHost));
    env->ReleaseFloatArrayElements(result, hostResult, 0);
    cudaFree(deviceArray);
}

/**
 *
 *  JNI functions for vector operations
 *
 */
extern "C"
{
    // Negate
    JNIEXPORT jobject JNICALL Java_brain_math_GpuVector_negate(JNIEnv *env, jobject obj)
    {
        // Retrieve the size field and values array from the vector
        jclass vectorClass = env->GetObjectClass(obj);
        jfieldID valuesField = env->GetFieldID(vectorClass, "values", "[F");
        jfloatArray jData = (jfloatArray)env->GetObjectField(obj, valuesField);
        jint size = env->GetArrayLength(jData);

        // Allocate GPU memory and copy data from the vector
        float *d_A = allocateJfloatArrayAndCopyToDevice(env, jData, size);
        float *d_C;
        checkCudaError(cudaMalloc(&d_C, size * sizeof(float)), "Failed to allocate output memory");

        // Launch kernel
        int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        negateKernel<<<gridSize, BLOCK_SIZE>>>(d_A, d_C, size);

        // Copy result back to host
        jfloatArray resultArray = copyDeviceToHostAndCreateArray(env, d_C, size);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_C);
        cudaDeviceReset();

        // Create and return new GpuVector with the result
        jobject resultVector = env->NewObject(vectorClass, env->GetMethodID(vectorClass, "<init>", "(I)V"), size);
        env->SetObjectField(resultVector, valuesField, resultArray);
        return resultVector;
    }

    // Add
    JNIEXPORT jobject JNICALL Java_brain_math_GpuVector_add(JNIEnv *env, jobject obj, jobject v)
    {
        // Retrieve the size field and values array from both vectors
        jclass vectorClass = env->GetObjectClass(obj);
        jfieldID sizeField = env->GetFieldID(vectorClass, "values", "[F");
        jint size = env->GetArrayLength((jfloatArray)env->GetObjectField(obj, sizeField));

        jfloatArray jData1 = (jfloatArray)env->GetObjectField(obj, sizeField);
        jfloatArray jData2 = (jfloatArray)env->GetObjectField(v, sizeField);

        // Allocate GPU memory and copy data from both vectors
        float *d_A = allocateJfloatArrayAndCopyToDevice(env, jData1, size);
        float *d_B = allocateJfloatArrayAndCopyToDevice(env, jData2, size);
        float *d_C;
        checkCudaError(cudaMalloc(&d_C, size * sizeof(float)), "Failed to allocate output memory");

        // Launch kernel
        int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        addKernel<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, size);

        // Copy result back to host
        jfloatArray resultArray = copyDeviceToHostAndCreateArray(env, d_C, size);

        // Free device memory
        cudaDeviceReset();

        // Create and return new GpuVector with the result
        jobject resultVector = env->NewObject(vectorClass, env->GetMethodID(vectorClass, "<init>", "(I)V"), size);
        env->SetObjectField(resultVector, sizeField, resultArray);
        return resultVector;
    }

    // Sub
    JNIEXPORT jobject JNICALL Java_brain_math_GpuVector_sub(JNIEnv *env, jobject obj, jobject v)
    {
        // Retrieve the size field and values array from both vectors
        jclass vectorClass = env->GetObjectClass(obj);
        jfieldID sizeField = env->GetFieldID(vectorClass, "values", "[F");
        jint size = env->GetArrayLength((jfloatArray)env->GetObjectField(obj, sizeField));

        jfloatArray jData1 = (jfloatArray)env->GetObjectField(obj, sizeField);
        jfloatArray jData2 = (jfloatArray)env->GetObjectField(v, sizeField);

        // Allocate GPU memory and copy data from both vectors
        float *d_A = allocateJfloatArrayAndCopyToDevice(env, jData1, size);
        float *d_B = allocateJfloatArrayAndCopyToDevice(env, jData2, size);
        float *d_C;
        checkCudaError(cudaMalloc(&d_C, size * sizeof(float)), "Failed to allocate output memory");

        // Launch kernel
        int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        subKernel<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, size);

        // Copy result back to host
        jfloatArray resultArray = copyDeviceToHostAndCreateArray(env, d_C, size);

        // Free device memory
        cudaDeviceReset();

        // Create and return new GpuVector with the result
        jobject resultVector = env->NewObject(vectorClass, env->GetMethodID(vectorClass, "<init>", "(I)V"), size);
        env->SetObjectField(resultVector, sizeField, resultArray);
        return resultVector;
    }

    // Multiply by Scalar
    JNIEXPORT jobject JNICALL Java_brain_math_GpuVector_mult__F(JNIEnv *env, jobject obj, jfloat factor)
    {
        // Retrieve the size field and values array from the vector
        jclass vectorClass = env->GetObjectClass(obj);
        jfieldID sizeField = env->GetFieldID(vectorClass, "values", "[F");
        jint size = env->GetArrayLength((jfloatArray)env->GetObjectField(obj, sizeField));

        jfloatArray jData = (jfloatArray)env->GetObjectField(obj, sizeField);

        // Allocate GPU memory and copy data from the vector
        float *d_A = allocateJfloatArrayAndCopyToDevice(env, jData, size);
        float *d_C;
        checkCudaError(cudaMalloc(&d_C, size * sizeof(float)), "Failed to allocate output memory");

        // Launch kernel
        int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        multScalarKernel<<<gridSize, BLOCK_SIZE>>>(d_A, factor, d_C, size);

        // Copy result back to host
        jfloatArray resultArray = copyDeviceToHostAndCreateArray(env, d_C, size);

        // Free device memory
        cudaDeviceReset();

        // Create and return new GpuVector with the result
        jobject resultVector = env->NewObject(vectorClass, env->GetMethodID(vectorClass, "<init>", "(I)V"), size);
        env->SetObjectField(resultVector, sizeField, resultArray);
        return resultVector;
    }

    // Multiply by Vector
    JNIEXPORT jobject JNICALL Java_brain_math_GpuVector_mult__Lbrain_math_Vector_2(JNIEnv *env, jobject obj, jobject v)
    {
        // Retrieve the size field and values array from both vectors
        jclass vectorClass = env->GetObjectClass(obj);
        jfieldID sizeField = env->GetFieldID(vectorClass, "values", "[F");
        jint size = env->GetArrayLength((jfloatArray)env->GetObjectField(obj, sizeField));

        jfloatArray jData1 = (jfloatArray)env->GetObjectField(obj, sizeField);
        jfloatArray jData2 = (jfloatArray)env->GetObjectField(v, sizeField);

        // Allocate GPU memory and copy data from both vectors
        float *d_A = allocateJfloatArrayAndCopyToDevice(env, jData1, size);
        float *d_B = allocateJfloatArrayAndCopyToDevice(env, jData2, size);
        float *d_C;
        checkCudaError(cudaMalloc(&d_C, size * sizeof(float)), "Failed to allocate output memory");

        // Launch kernel
        int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        multVectorKernel<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_C, size);

        // Copy result back to host
        jfloatArray resultArray = copyDeviceToHostAndCreateArray(env, d_C, size);

        // Free device memory
        cudaDeviceReset();

        // Create and return new GpuVector with the result
        jobject resultVector = env->NewObject(vectorClass, env->GetMethodID(vectorClass, "<init>", "(I)V"), size);
        env->SetObjectField(resultVector, sizeField, resultArray);
        return resultVector;
    }

    JNIEXPORT jobject JNICALL Java_brain_math_GpuVector_mult__Lbrain_math_Matrix_2(JNIEnv *env, jobject obj, jobject m)
    {
        // Retrieve the size field and values array from both matrices
        jclass matrixClass = env->GetObjectClass(obj);
        jfieldID valuesField = env->GetFieldID(matrixClass, "values", "[Lbrain/math/Vector;");
        jobjectArray jData1 = (jobjectArray)env->GetObjectField(obj, valuesField);
        jobjectArray jData2 = (jobjectArray)env->GetObjectField(m, valuesField);

        jint A_rows = env->GetArrayLength(jData1);
        jint A_cols = env->GetArrayLength((jfloatArray)env->GetObjectArrayElement(jData1, 0));
        jint B_rows = env->GetArrayLength(jData2);
        jint B_cols = env->GetArrayLength((jfloatArray)env->GetObjectArrayElement(jData2, 0));

        if (A_cols != B_rows)
        {
            jclass illegalArgumentException = env->FindClass("java/lang/IllegalArgumentException");
            env->ThrowNew(illegalArgumentException, "Matrix column amount must match");
            return nullptr;
        }

        // Allocate GPU memory and copy data from both matrices
        float *d_A = allocateJobjectArrayAndCopyToDevice(env, jData1, A_rows, A_cols);
        float *d_B = allocateJobjectArrayAndCopyToDevice(env, jData2, B_rows, B_cols);
        float *d_C;
        checkCudaError(cudaMalloc(&d_C, A_rows * B_cols * sizeof(float)), "Failed to allocate GPU memory for result");

        // Launch kernel
        dim3 blockSize(16, 16);
        dim3 gridSize((B_cols + blockSize.x - 1) / blockSize.x, (A_rows + blockSize.y - 1) / blockSize.y);
        matMulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, A_rows, A_cols, B_cols);

        // Copy result back to host
        float *h_C = new float[A_rows * B_cols];
        cudaMemcpy(h_C, d_C, A_rows * B_cols * sizeof(float), cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);

        // Reset the device
        cudaDeviceReset();

        // Create a new GpuVector object for each row in the resulting matrix
        jclass vectorClass = env->FindClass("brain/math/GpuVector");
        jmethodID vectorConstructor = env->GetMethodID(vectorClass, "<init>", "(I)V");
        jobjectArray resultArray = env->NewObjectArray(A_rows, vectorClass, nullptr);

        for (int i = 0; i < A_rows; ++i)
        {
            jfloatArray rowArray = env->NewFloatArray(B_cols);
            env->SetFloatArrayRegion(rowArray, 0, B_cols, h_C + i * B_cols);
            jobject rowVector = env->NewObject(vectorClass, vectorConstructor, B_cols);
            env->SetObjectField(rowVector, valuesField, rowArray);
            env->SetObjectArrayElement(resultArray, i, rowVector);
        }

        delete[] h_C;

        // Create and return new GpuMatrix with the result
        jobject resultMatrix = env->NewObject(matrixClass, env->GetMethodID(matrixClass, "<init>", "(II)V"), B_cols, A_rows);
        env->SetObjectField(resultMatrix, valuesField, resultArray);
        return resultMatrix;
    }

    // Division by Scalar
    JNIEXPORT jobject JNICALL Java_brain_math_GpuVector_div(JNIEnv *env, jobject obj, jfloat divisor)
    {
        return Java_brain_math_GpuVector_mult__F(env, obj, 1.0f / divisor); // Reuse scalar mult method with reciprocal divisor
    }

    // Dot Product
    JNIEXPORT jfloat JNICALL Java_brain_math_GpuVector_dot(JNIEnv *env, jobject obj, jobject v)
    {
        // Retrieve the size field and values array from both vectors
        jclass vectorClass = env->GetObjectClass(obj);
        jfieldID sizeField = env->GetFieldID(vectorClass, "values", "[F");
        jint size = env->GetArrayLength((jfloatArray)env->GetObjectField(obj, sizeField));

        jfloatArray jData1 = (jfloatArray)env->GetObjectField(obj, sizeField);
        jfloatArray jData2 = (jfloatArray)env->GetObjectField(v, sizeField);

        // Allocate GPU memory and copy data from both vectors
        float *d_A = allocateJfloatArrayAndCopyToDevice(env, jData1, size);
        float *d_B = allocateJfloatArrayAndCopyToDevice(env, jData2, size);
        float *d_result;
        checkCudaError(cudaMalloc(&d_result, sizeof(float)), "Failed to allocate result memory");

        checkCudaError(cudaMemset(d_result, 0, sizeof(float)), "Failed to zero out result");

        // Launch kernel
        int gridSize = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        dotKernel<<<gridSize, BLOCK_SIZE>>>(d_A, d_B, d_result, size);

        // Copy result back to host
        float result;
        checkCudaError(cudaMemcpy(&result, d_result, sizeof(float), cudaMemcpyDeviceToHost), "Failed to copy result to host");

        // Free device memory
        cudaDeviceReset();

        return result;
    }
}