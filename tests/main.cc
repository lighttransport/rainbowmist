#include <cstdlib>
#include <cstdio>

#define CATCH_CONFIG_RUNNER
#include "catch.hpp"

#include "cuew.h"

#include "EasyCL.h"

#include "cupp11.h"

// ------------
#include "simple_add.kernel"
// ------------

using namespace Catch;
using namespace easycl;

static bool hasCUDA = false;
static bool hasOpenCL = false;

static std::string kOpenCLCompileOptions = "-I ../ -I ../../ -D OPENCL";
static std::string kCUDACompileOptions = "--include-path=../../";

std::string LoadFile(const std::string &filename)
{
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

TEST_CASE("CUDA initialize", "[cuda][!mayfail]")
{
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
  compiler_options.push_back(kCUDACompileOptions);
  auto build_status = program.Build(device, compiler_options);
  if (build_status != CLCudaAPI::BuildStatus::kSuccess) {
    auto message = program.GetBuildInfo(device);
    std::cerr << " > Compiler error(s)/warning(s) found: " << std::endl << message << std::endl;
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

  auto kernel = CLCudaAPI::Kernel(program, "simple_add_float2");
  kernel.SetArgument(0, dev_ret);
  kernel.SetArgument(1, dev_a);
  kernel.SetArgument(2, dev_b);

  std::vector<size_t> global(1, 1);
  std::vector<size_t> local(1, 1);

  kernel.Launch(queue, global, local, event.pointer());
  queue.Finish(event);

  dev_ret.Read(queue, 2, ret);

  REQUIRE( ret[0] == Approx(4) );
  REQUIRE( ret[1] == Approx(6.6) );

}

// -----------------------------------------------

TEST_CASE("OCL simple add float2", "[opencl]")
{
  EasyCL *cl = EasyCL::createForFirstGpu();
  CLKernel *kernel = cl->buildKernel("../simple_add.kernel", "simple_add_float2", kOpenCLCompileOptions);
  REQUIRE(kernel != nullptr);

  float ret[2];
  float a[2], b[2];

  a[0] = 1;
  a[1] = 2.1;

  b[0] = 3;
  b[1] = 4.5;

  kernel->out(2, reinterpret_cast<float *>(&ret));
  kernel->in(2, a);
  kernel->in(2, b);

  kernel->run_1d(1, 1);

  REQUIRE( ret[0] == Approx(4) );
  REQUIRE( ret[1] == Approx(6.6) );
}

// -----------------------------------------------

TEST_CASE("simple add float2", "[cpp11]")
{
  float2 ret;
  float2 a, b;

  a = make_float2(1, 2.1);
  b = make_float2(3, 4.5);

  simple_add_float2(&ret, &a, &b);

  REQUIRE( ret.x == Approx(4) );
  REQUIRE( ret.y == Approx(6.6) );
}

int main(int argc, char **argv)
{
  {
    int ret = cuewInit();
    if (ret == CUEW_SUCCESS) {
      hasCUDA = true;
    }
  }

  {
    hasOpenCL = EasyCL::isOpenCLAvailable();
  }

  
  int result = Catch::Session().run(argc, argv);
  return ( result < 0xff ) ? result : 0xff;
}
