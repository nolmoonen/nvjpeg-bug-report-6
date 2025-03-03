#include <cuda_runtime.h>
#include <nvjpeg.h>

#include <fstream>
#include <iostream>
#include <random>
#include <sstream>
#include <stdint.h>
#include <vector>

#define CUDA_CALL(call)                                                        \
  do {                                                                         \
    cudaError_t res_ = call;                                                   \
    if (cudaSuccess != res_) {                                                 \
      std::cout << "CUDA error at " << __FILE__ << ":" << __LINE__             \
                << "code=" << static_cast<unsigned int>(res_) << "("           \
                << cudaGetErrorString(res_) << ") \"" << #call << "\"\n";      \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

#define NVJPEG_CALL(call)                                                      \
  do {                                                                         \
    nvjpegStatus_t res_ = call;                                                \
    if (NVJPEG_STATUS_SUCCESS != res_) {                                       \
      std::cout << "nvJPEG error at " << __FILE__ << ":" << __LINE__           \
                << " code=" << static_cast<unsigned int>(res_) << "\n";        \
      std::exit(EXIT_FAILURE);                                                 \
    }                                                                          \
  } while (0)

// produce a jpeg image in `jpeg` with pixel values are random based on `seed`
void encode_random_image(std::vector<uint8_t> &jpeg, uint8_t *h_img,
                         uint8_t *d_img, nvjpegImage_t &nv_image, int size_x,
                         int size_y, int num_components, size_t num_bytes,
                         int seed) {
  std::minstd_rand gen(seed);
  std::uniform_int_distribution<int> dist(0, 255);
  for (size_t i = 0; i < num_bytes; ++i) {
    h_img[i] = dist(gen);
  }

  cudaStream_t stream = nullptr;

  nvjpegHandle_t nv_handle;
  NVJPEG_CALL(nvjpegCreateSimple(&nv_handle));

  nvjpegEncoderState_t nv_enc_state;
  NVJPEG_CALL(nvjpegEncoderStateCreate(nv_handle, &nv_enc_state, stream));

  nvjpegEncoderParams_t nv_enc_params;
  NVJPEG_CALL(nvjpegEncoderParamsCreate(nv_handle, &nv_enc_params, stream));

  NVJPEG_CALL(nvjpegEncoderParamsSetSamplingFactors(nv_enc_params,
                                                    NVJPEG_CSS_444, stream));
  NVJPEG_CALL(nvjpegEncoderParamsSetOptimizedHuffman(nv_enc_params, 0, stream));
  NVJPEG_CALL(nvjpegEncoderParamsSetEncoding(
      nv_enc_params, NVJPEG_ENCODING_BASELINE_DCT, stream));
  NVJPEG_CALL(nvjpegEncoderParamsSetQuality(nv_enc_params, 90, stream));

  CUDA_CALL(cudaMemcpy(d_img, h_img, num_bytes, cudaMemcpyHostToDevice));

  NVJPEG_CALL(nvjpegEncodeImage(nv_handle, nv_enc_state, nv_enc_params,
                                &nv_image, NVJPEG_INPUT_RGB, size_x, size_y,
                                stream));
  CUDA_CALL(cudaStreamSynchronize(stream));

  size_t length{};
  NVJPEG_CALL(nvjpegEncodeRetrieveBitstream(nv_handle, nv_enc_state, nullptr,
                                            &length, stream));
  jpeg.resize(length);
  NVJPEG_CALL(nvjpegEncodeRetrieveBitstream(
      nv_handle, nv_enc_state, reinterpret_cast<unsigned char *>(jpeg.data()),
      &length, stream));
  CUDA_CALL(cudaStreamSynchronize(stream));

  NVJPEG_CALL(nvjpegEncoderParamsDestroy(nv_enc_params));
  NVJPEG_CALL(nvjpegEncoderStateDestroy(nv_enc_state));
  NVJPEG_CALL(nvjpegDestroy(nv_handle));
}

void write_file(const char *filename, const uint8_t *jpeg, size_t length) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  file.write(reinterpret_cast<const char *>(jpeg), length);
}

void write_ppm(const char *filename, int size_x, int size_y, int num_components,
               const std::vector<uint8_t> &h_img) {
  std::ofstream file(filename, std::ios::out | std::ios::binary);
  file << "P6\n" << size_x << " " << size_y << "\n255\n";
  const size_t component_size = size_x * size_y;
  for (int i = 0; i < component_size; ++i) {
    file << h_img[0 * component_size + i] << h_img[1 * component_size + i]
         << h_img[2 * component_size + i];
  }
}

int main(int argc, char *argv[]) {
  float truncated_fraction{};
  if (argc < 2) {
    std::cout << "usage: repro <truncated_fraction>\n";
    return EXIT_FAILURE;
  }

  std::istringstream ss(argv[1]);
  if (!(ss >> truncated_fraction)) {
    std::cout << "failed to parse\n";
    return EXIT_FAILURE;
  }

  const int size_x = 256;
  const int size_y = 256;
  const int num_components = 3;
  const int component_size = size_x * size_y;
  const size_t num_bytes = num_components * component_size;
  std::vector<uint8_t> h_img(num_bytes);

  uint8_t *d_img = nullptr;
  CUDA_CALL(cudaMalloc(&d_img, num_bytes));

  nvjpegImage_t nv_image{};
  int off = 0;
  for (int c = 0; c < num_components; ++c) {
    nv_image.channel[c] = d_img + off;
    nv_image.pitch[c] = size_x;
    off += size_x * size_y;
  }

  nvjpegHandle_t nv_handle;
  NVJPEG_CALL(nvjpegCreateSimple(&nv_handle));

  nvjpegJpegState_t jpeg_state{};
  NVJPEG_CALL(nvjpegJpegStateCreate(nv_handle, &jpeg_state));

  nvjpegJpegDecoder_t nvjpeg_decoder{};
  NVJPEG_CALL(nvjpegDecoderCreate(nv_handle, NVJPEG_BACKEND_GPU_HYBRID,
                                  &nvjpeg_decoder));
  nvjpegJpegState_t nvjpeg_decoupled_state;
  NVJPEG_CALL(nvjpegDecoderStateCreate(nv_handle, nvjpeg_decoder,
                                       &nvjpeg_decoupled_state));

  nvjpegBufferPinned_t pinned_buffer;
  NVJPEG_CALL(nvjpegBufferPinnedCreate(nv_handle, nullptr, &pinned_buffer));
  nvjpegBufferDevice_t device_buffer;
  NVJPEG_CALL(nvjpegBufferDeviceCreate(nv_handle, nullptr, &device_buffer));

  nvjpegJpegStream_t jpeg_stream;
  NVJPEG_CALL(nvjpegJpegStreamCreate(nv_handle, &jpeg_stream));

  nvjpegDecodeParams_t nvjpeg_decode_params;
  NVJPEG_CALL(nvjpegDecodeParamsCreate(nv_handle, &nvjpeg_decode_params));

  NVJPEG_CALL(
      nvjpegStateAttachPinnedBuffer(nvjpeg_decoupled_state, pinned_buffer));
  NVJPEG_CALL(
      nvjpegStateAttachDeviceBuffer(nvjpeg_decoupled_state, device_buffer));
  NVJPEG_CALL(nvjpegDecodeParamsSetOutputFormat(nvjpeg_decode_params,
                                                NVJPEG_OUTPUT_RGB));

  auto decode = [&](uint8_t *jpeg, size_t jpeg_size) {
    cudaStream_t stream = nullptr;

    // since nvjpeg may choose to not write all `d_img` values, set them to zero
    //   to be sure
    CUDA_CALL(cudaMemset(d_img, 0, num_bytes));

    NVJPEG_CALL(nvjpegJpegStreamParse(nv_handle,
                                      reinterpret_cast<unsigned char *>(jpeg),
                                      jpeg_size, 0, 0, jpeg_stream));

    NVJPEG_CALL(nvjpegDecodeJpegHost(nv_handle, nvjpeg_decoder,
                                     nvjpeg_decoupled_state,
                                     nvjpeg_decode_params, jpeg_stream));

    NVJPEG_CALL(nvjpegDecodeJpegTransferToDevice(nv_handle, nvjpeg_decoder,
                                                 nvjpeg_decoupled_state,
                                                 jpeg_stream, stream));

    NVJPEG_CALL(nvjpegDecodeJpegDevice(
        nv_handle, nvjpeg_decoder, nvjpeg_decoupled_state, &nv_image, stream));

    CUDA_CALL(cudaStreamSynchronize(stream));
    CUDA_CALL(
        cudaMemcpy(h_img.data(), d_img, num_bytes, cudaMemcpyDeviceToHost));
  };

  std::vector<uint8_t> jpeg;

  // step 1: create a random image and decode it
  encode_random_image(jpeg, h_img.data(), d_img, nv_image, size_x, size_y,
                      num_components, num_bytes, 43 /* seed */);
  write_file("test_43.jpg", jpeg.data(), jpeg.size());
  decode(jpeg.data(), jpeg.size());

  // step 2: create a different random image, truncate it and decode it with the
  //   same state as the previous step
  encode_random_image(jpeg, h_img.data(), d_img, nv_image, size_x, size_y,
                      num_components, num_bytes, 44 /* seed */);
  write_file("test_44_0.jpg", jpeg.data(), jpeg.size());
  const size_t length_truncated_44_0 = jpeg.size() * truncated_fraction;
  write_file("test_44_trunc_0.jpg", jpeg.data(), length_truncated_44_0);
  decode(jpeg.data(), length_truncated_44_0);
  write_ppm("test_44_trunc_decoded_0.ppm", size_x, size_y, num_components,
            h_img);

  // step 3: create a third random image different from the previous steps, and
  //   decode it with the same state
  encode_random_image(jpeg, h_img.data(), d_img, nv_image, size_x, size_y,
                      num_components, num_bytes, 45 /* seed */);
  write_file("test_45.jpg", jpeg.data(), jpeg.size());
  decode(jpeg.data(), jpeg.size());

  // step 4: encode the exact image as in step 2, decode it and output the
  //   result
  encode_random_image(jpeg, h_img.data(), d_img, nv_image, size_x, size_y,
                      num_components, num_bytes, 44 /* seed */);
  write_file("test_44_1.jpg", jpeg.data(), jpeg.size());
  const size_t length_truncated_44_1 = jpeg.size() * truncated_fraction;
  write_file("test_44_trunc_1.jpg", jpeg.data(), length_truncated_44_1);
  decode(jpeg.data(), length_truncated_44_1);
  write_ppm("test_44_trunc_decoded_1.ppm", size_x, size_y, num_components,
            h_img);

  NVJPEG_CALL(nvjpegDecodeParamsDestroy(nvjpeg_decode_params));
  NVJPEG_CALL(nvjpegJpegStreamDestroy(jpeg_stream));
  NVJPEG_CALL(nvjpegBufferPinnedDestroy(pinned_buffer));
  NVJPEG_CALL(nvjpegBufferDeviceDestroy(device_buffer));
  NVJPEG_CALL(nvjpegJpegStateDestroy(nvjpeg_decoupled_state));
  NVJPEG_CALL(nvjpegDecoderDestroy(nvjpeg_decoder));
  NVJPEG_CALL(nvjpegJpegStateDestroy(jpeg_state));
  NVJPEG_CALL(nvjpegDestroy(nv_handle));

  CUDA_CALL(cudaFree(d_img));
}
