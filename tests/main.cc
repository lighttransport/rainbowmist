#include <cstdio>
#include <cstdlib>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#ifdef __clang__
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wreserved-id-macro"
#pragma clang diagnostic ignored "-Wundef"
#pragma clang diagnostic ignored "-Wunused-parameter"
#pragma clang diagnostic ignored "-Wmissing-noreturn"
#pragma clang diagnostic ignored "-Wshadow-field-in-constructor"
#pragma clang diagnostic ignored "-Wsign-conversion"
#pragma clang diagnostic ignored "-Wshorten-64-to-32"
#pragma clang diagnostic ignored "-Wfloat-conversion"
#pragma clang diagnostic ignored "-Wweak-vtables"
#endif

#include "cuew.h"

#include "EasyCL.h"

#include "cupp11.h"

#ifdef __clang__
#pragma clang diagnostic pop
#endif

#ifdef __clang__
#pragma clang diagnostic ignored "-Wexit-time-destructors"
#pragma clang diagnostic ignored "-Wundefined-func-template"
#endif

// ------------
#include "alignment.kernel"
#include "simple_add.kernel"
// ------------

using namespace Catch;
using namespace easycl;

static bool hasCUDA = false;
static bool hasOpenCL = false;

static std::string kOpenCLCompileOptions = "-I ../ -I ../../ -D OPENCL";

static std::vector<std::string> kCUDACompileOptions = {"--include-path=../../"};

#if !defined(__APPLE__)
static std::string LoadFile(const std::string &filename) {
  std::ifstream ifs(filename);
  if (!ifs) {
    std::cerr << "failed to open file : " << filename << std::endl;
  }

  ifs.seekg(0, ifs.end);
  int length = static_cast<int>(ifs.tellg());
  if (length < 4) {
    return std::string();
  }

  ifs.seekg(0, ifs.beg);

  std::vector<unsigned char> buf;
  buf.resize(length);

  ifs.read(reinterpret_cast<char *>(buf.data()), length);

  std::string s(buf.data(), buf.data() + length);

  return s;
}

TEST_CASE("CUDA initialize", "[cuda]") {
  auto platform = CLCudaAPI::Platform(0);
  auto device = CLCudaAPI::Device(platform, 0);
  auto context = CLCudaAPI::Context(device);
  auto queue = CLCudaAPI::Queue(context, device);
  auto event = CLCudaAPI::Event();

  std::string program_string = LoadFile("../simple_add.kernel");
  std::cout << program_string << std::endl;
  REQUIRE(!program_string.empty());

  auto program = CLCudaAPI::Program(context, std::move(program_string));
  std::vector<std::string> compiler_options;
  compiler_options = kCUDACompileOptions;
  auto build_status = program.Build(device, compiler_options);
  if (build_status != CLCudaAPI::BuildStatus::kSuccess) {
    auto message = program.GetBuildInfo(device);
    std::cerr << " > Compiler error(s)/warning(s) found: " << std::endl
              << message << std::endl;
  }
  REQUIRE(build_status == CLCudaAPI::BuildStatus::kSuccess);

  float ret[2];
  float a[2], b[2];

  a[0] = 1;
  a[1] = 2.1;

  b[0] = 3;
  b[1] = 4.5;

  auto dev_a = CLCudaAPI::Buffer<float>(context, queue, a, a + 2);
  auto dev_b = CLCudaAPI::Buffer<float>(context, queue, b, b + 2);
  auto dev_ret = CLCudaAPI::Buffer<float>(context, queue, ret, ret + 2);

  auto kernel = CLCudaAPI::Kernel(program, "simple_add_vec2");
  kernel.SetArgument(0, dev_ret);
  kernel.SetArgument(1, dev_a);
  kernel.SetArgument(2, dev_b);

  std::vector<size_t> global(1, 1);
  std::vector<size_t> local(1, 1);

  kernel.Launch(queue, global, local, event.pointer());
  queue.Finish(event);

  dev_ret.Read(queue, 2, ret);

  REQUIRE(ret[0] == Approx(4));
  REQUIRE(ret[1] == Approx(6.6));
}

TEST_CASE("CUDA datasize", "[cuda]") {
  auto platform = CLCudaAPI::Platform(0);
  auto device = CLCudaAPI::Device(platform, 0);
  auto context = CLCudaAPI::Context(device);
  auto queue = CLCudaAPI::Queue(context, device);
  auto event = CLCudaAPI::Event();

  std::string program_string = LoadFile("../alignment.kernel");
  std::cout << program_string << std::endl;
  REQUIRE(!program_string.empty());

  auto program = CLCudaAPI::Program(context, std::move(program_string));
  std::vector<std::string> compiler_options;
  compiler_options = kCUDACompileOptions;
  auto build_status = program.Build(device, compiler_options);
  if (build_status != CLCudaAPI::BuildStatus::kSuccess) {
    auto message = program.GetBuildInfo(device);
    std::cerr << " > Compiler error(s)/warning(s) found: " << std::endl
              << message << std::endl;
  }
  REQUIRE(build_status == CLCudaAPI::BuildStatus::kSuccess);

  int ret[3];

  auto dev_ret = CLCudaAPI::Buffer<int>(context, queue, ret, ret + 3);

  auto kernel = CLCudaAPI::Kernel(program, "alignment_test");
  kernel.SetArgument(0, dev_ret);

  std::vector<size_t> global(1, 1);
  std::vector<size_t> local(1, 1);

  kernel.Launch(queue, global, local, event.pointer());
  queue.Finish(event);

  dev_ret.Read(queue, 3, ret);

  REQUIRE(ret[0] == 12);
  REQUIRE(ret[1] == 24);
  REQUIRE(ret[2] == 32);
}
#endif

// -----------------------------------------------

TEST_CASE("OCL simple add vec2", "[opencl]") {
  EasyCL *cl = EasyCL::createForFirstGpu();
  CLKernel *kernel = cl->buildKernel("../simple_add.kernel", "simple_add_vec2",
                                     kOpenCLCompileOptions);
  REQUIRE(kernel != nullptr);

  float ret[2];
  float a[2], b[2];

  a[0] = 1;
  a[1] = 2.1f;

  b[0] = 3;
  b[1] = 4.5f;

  kernel->out(2, reinterpret_cast<float *>(&ret));
  kernel->in(2, a);
  kernel->in(2, b);

  kernel->run_1d(1, 1);

  REQUIRE(ret[0] == Approx(4));
  REQUIRE(ret[1] == Approx(6.6f));
}

TEST_CASE("OCL datasize", "[opencl]") {
  EasyCL *cl = EasyCL::createForFirstGpu();
  CLKernel *kernel = cl->buildKernel("../alignment.kernel", "alignment_test",
                                     kOpenCLCompileOptions);
  REQUIRE(kernel != nullptr);

  int ret[3];

  kernel->out(3, reinterpret_cast<int *>(&ret));

  kernel->run_1d(1, 1);

  REQUIRE(ret[0] == 16);  // sizeof(vec3)
  REQUIRE(ret[1] == 32);  // sizeof(Ray)
  REQUIRE(ret[2] == 32);  // sizeof(Ray16)
}

// -----------------------------------------------

TEST_CASE("simple add vec2", "[cpp11]") {
  vec2 ret;
  vec2 a, b;

  a = make_vec2(1, 2.1f);
  b = make_vec2(3, 4.5f);

  simple_add_vec2(&ret, &a, &b);

  REQUIRE(ret.x == Approx(4));
  REQUIRE(ret.y == Approx(6.6f));
}

TEST_CASE("datasize", "[cpp11]") {

  int ret[3];

  alignment_test(ret);

  REQUIRE(ret[0] == 12);
  REQUIRE(ret[1] == 24);
  REQUIRE(ret[2] == 32);

}

int main(int argc, char **argv) {
  std::vector<char *> local_argv;

  char *cuda_exclude_opt = nullptr;
  char *opencl_exclude_opt = nullptr;

  bool runCUDA = true;
  bool runOpenCL = true;

  for (int i = 0; i < argc; i++) {
    local_argv.push_back(argv[i]);

    std::cout << argv[i] << std::endl;
    if (strcmp("exclude:[cuda]", argv[i]) == 0) {
      runCUDA = false;
    }
    if (strcmp("exclude:[opencl]", argv[i]) == 0) {
      runCUDA = false;
    }
  }

  if (runCUDA) {
    // NOTE: `cuewInit()` may seg fault on some situation even when using try-catch exception handling.
    // (e.g. installing cuda toolkit on Intel or AMD only GPU devices)
    std::cout << "Initialize CUDA(You can skip CUDA test using `exclude:[cuda]' argument if you face a CUDA initialization problem)." << std::endl;
    std::cout << std::flush;
    try {
      int ret = cuewInit(CUEW_INIT_CUDA | CUEW_INIT_NVRTC);
      if (ret == CUEW_SUCCESS) {
        hasCUDA = true;

        printf("NVCC path    : %s\n", cuewCompilerPath());
        printf("NVCC version : %d\n", cuewCompilerVersion());

        if (nvrtcVersion) {
          int major, minor;
          nvrtcVersion(&major, &minor);
          std::cout << "NVRTC version : " << major << "." << minor << std::endl;
        } else {
          // Guess in 32bit mode. NVRTC is not available on Windows/Linux 32bit.
          std::cerr << "NVRTC not available.";
          exit(-1123);
        }

        // check if actual CUDA device is available.
        {
          bool failed = false;

          if (CUDA_SUCCESS != cuInit(0)) {
            std::cerr << "cuInit failed." << std::endl;
            failed = true;
          } else {
            int num_devices = 0;
            CUresult cu_ret = cuDeviceGetCount(&num_devices);
            if (cu_ret != CUDA_SUCCESS) {
              std::cerr << "cuDeviceGetCound err : " << cu_ret << std::endl;
              failed = true;
            }

            std::cout << "# of CUDA devices : " << num_devices << std::endl;

            if (num_devices == 0) {
              std::cerr << "CUDA capable device not available." << std::endl;
              failed = true;
            }
          }

          if (failed) {
            cuda_exclude_opt = strdup("exclude:[cuda]");
            local_argv.push_back(cuda_exclude_opt);
          }
        }
      } else {
        std::cerr << "CUDA not available." << std::endl;
        cuda_exclude_opt = strdup("exclude:[cuda]");
        local_argv.push_back(cuda_exclude_opt);
      }
    } catch(std::exception &e) {
      std::cerr << "Failed to Initialize CUDA. reason = " << e.what() << std::endl;
      cuda_exclude_opt = strdup("exclude:[cuda]");
      local_argv.push_back(cuda_exclude_opt);
    }
  } else {
    std::cout << "Skip testing with CUDA code path..." << std::endl;
  }

  if (runOpenCL) {
    std::cout << "Initialize OpenCL(You can skip OpenCL test using `exclude:[opencl]' argument if you face a OpenCL initialization problem)." << std::endl;
    hasOpenCL = EasyCL::isOpenCLAvailable();

    if (hasOpenCL) {
    } else {
      std::cerr << "OpenCL not available." << std::endl;
      opencl_exclude_opt = strdup("exclude:[opencl]");
      local_argv.push_back(opencl_exclude_opt);
    }
  } else {
    std::cout << "Skip running tests of OpenCL code path..." << std::endl;
  }

  int local_argc = int(local_argv.size());

  int result = Catch::Session().run(local_argc, local_argv.data());

  if (cuda_exclude_opt) {
    free(cuda_exclude_opt);
  }

  if (opencl_exclude_opt) {
    free(opencl_exclude_opt);
  }

  return (result < 0xff) ? result : 0xff;
}
