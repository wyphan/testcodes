#ifndef HELPERS_HIP_HPP
#define HELPERS_HIP_HPP 1

// System includes
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <limits>
#include <cctype>
#include <string>
#include <vector>
#include <algorithm>
#include <locale>

// HIP includes
#include <hip/hip_runtime.h>

// Macros
#define isNon0(foo) (foo != 0) ? true : false
#define YesNo(foo) ((foo) ? "Yes" : "No")
#define toKiB(foo) (double)foo / (double)(1<<10)
#define toMiB(foo) (double)foo / (double)(1<<20)
#define toGiB(foo) (double)foo / (double)(1<<30)

////////////////////////////////////////////////////////////////////////////////
// These three functions are taken from https://stackoverflow.com/a/217605/

// trim from start (in place)
static inline void ltrim(std::string &s) {
  s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](unsigned char ch) {
							return !std::isspace(ch);
						      }));
}

// trim from end (in place)
static inline void rtrim(std::string &s) {
  s.erase(std::find_if(s.rbegin(), s.rend(), [](unsigned char ch) {
					       return !std::isspace(ch);
					     }).base(), s.end());
}

// trim from both ends (in place)
static inline void trim(std::string &s) {
  ltrim(s);
  rtrim(s);
}

////////////////////////////////////////////////////////////////////////////////

// Struct to hold device id and properties
struct device_t {

  // GPU ID as detected
  int id;

  // Builtin structs
  hipDeviceProp_t prop;
  hipDeviceArch_t flags;

  // General info
  std::string pci;
  std::string name;
  std::string arch;
  std::string codename;
  std::string hip_cc;
  bool is_multigpu;
  bool is_apu;
  int rev;

  // Execution params
  int cu_count;
  int cuclk;
  int instclk;
  // int thr_per_cu;
  // int reg_per_blk;

  // Compute params
  int warpsz;
  int thr_per_blk;
  int max_thr[3];
  int max_grid[3];
  
  // Memory params
  int memclk; // kHz
  int bussz; // bits
  size_t global_memsz; // bytes
  // size_t l2_memsz; // bytes
  size_t shared_memsz; // bytes
  size_t shared_per_cu; // bytes
  size_t shared_per_blk; // bytes
  size_t pitch; // bytes
  bool ecc;
  bool large_bar;
  bool managed_mem;
  bool managed_host;
  bool managed_concur;
  bool maphost;
  bool pageable;
  bool pageable_host;

  // Execution control
  bool concur;
  bool coop;
  bool coop_multi;

};

// Vector for GPU names with the same arch
struct strvec_t {
  int nelems;
  std::vector<std::string> elem;
};

// Struct for each GPU arch
struct gpurow_t {
  std::string arch;
  std::string codename;
  strvec_t gpuname;
};

// Get number of GPUs, return -1 for failure
int get_ngpus () {
  int ngpus = 0;
  hipError_t stat;
  stat = hipGetDeviceCount(&ngpus);
  if (stat != hipSuccess) {
    std::cout << "Error: get_ngpus returned error " << stat << std::endl;
    std::cout << hipGetErrorString(stat) << std::endl;
    ngpus = -1;
  }
  return ngpus;
}

// Get GPU info, return -1 for select failure, -2 for query failure
int get_gpu_info (const int id, device_t& gpu) {
  hipError_t stat;

  // Select GPU
  stat = hipSetDevice(id);
  if (stat != hipSuccess) {
    std::cout << "Error: get_gpu_info cannot select device " << id
	      << ", error " << stat << std::endl;
    std::cout << hipGetErrorString(stat) << std::endl;
    return -1;
  }

  // Extract info
  stat = hipGetDeviceProperties(&gpu.prop, id);
  if (stat != hipSuccess) {
    std::cout << "Error: get_gpu_info cannot query device " << id
	      << ", error " << stat << std::endl;
    std::cout << hipGetErrorString(stat) << std::endl;
    return -2;
  }

  // Extract arch flags
  gpu.flags = gpu.prop.arch;

  return 0;
}

// Read GPU table
int read_table (const std::string tabfile,
		std::vector<gpurow_t>& gputable) {
  // Open table file
  std::ifstream f(tabfile);

  // Separators
  char comment = '#';
  char field_sep = ':';
  char array_sep = '/';
  char endl = '\n';

  // Parse table file
  char in = '0';
  while (! (in == EOF)) {
    in = f.peek();
    if (in == comment) {

      // Ignore lines that start with '#'
      f.ignore(std::numeric_limits<std::streamsize>::max(), endl);

    } else {

      // Read table file line-by-line
      const int maxbuf = 256;
      char buf[maxbuf];
      f.getline(buf, maxbuf, endl);
      std::string line = buf;
      int len = line.length();

      // Allocate struct for a single GPU arch
      gpurow_t gpufam = {};

      // Extract GCN arch name
      int pos1 = line.find_first_of(field_sep, 0);
      gpufam.arch = line.substr(0, pos1 - 1);
      trim(gpufam.arch);

      // Extract codename
      int pos2 = line.find_first_of(field_sep, pos1 + 1);
      gpufam.codename = line.substr(pos1 + 1, pos2 - pos1 - 1);
      trim(gpufam.codename);
      
      // Extract possible GPUs for that arch
      int pos3 = pos2 + 1;
      while (pos3 < len) {
	int pos4;
	if (line.find_first_of(array_sep, pos3) != std::string::npos) {
	  pos4 = line.find_first_of(array_sep, pos3);
	} else {
	  pos4 = len + 1;
	}
	std::string gpuname = line.substr(pos3 + 1, pos4 - pos3 - 1);
	trim(gpuname);
	gpufam.gpuname.elem.push_back(gpuname);
	pos3 = pos4 + 1;
      }
      gpufam.gpuname.nelems = gpufam.gpuname.elem.size();

      // Save row for GPU arch
      gputable.push_back(gpufam);

    } // in
  } // while

  f.close();
  return 0;
}

// Match the detected GPU arch with the GPU table database
int identify_gpu (const std::string arch,
		  int& family_idx, int& gpuname_idx,
		  const std::vector<gpurow_t> gputable) {

  // Extract just the GCN arch (discarding flags)
  char field_sep = ':';
  int pos = arch.find_first_of(field_sep);
  std::string archname = arch.substr(0, pos);

  // Loop through each GCN arch in the database
  bool found = false;
  for (int i = 0; i < gputable.size(); i++) {
    int nmatch = 0;
    int len = gputable[i].arch.size();
    for (int c = 0; c < len; c++) {
      if (archname[c] == gputable[i].arch[c]) {
	nmatch++;
      } // if
    } // c
    if (nmatch == len) {
      found = true;
    }
    if (found) {
      family_idx = i;
      break;
    }
  } // i

  // TODO: Match GPU based on lspci output
  //       For now, just emit the first one
  gpuname_idx = 0;

  if (! found) {
    std::cout << "GCN arch " << archname << " not found in database!" << std::endl;
    family_idx = -1;
    gpuname_idx = -1;
  }

  return 0;
}

// Fill in GPU data to display
int populate_gpu_fields (const int id, device_t& gpu,
			 const std::vector<gpurow_t> gputable) {

  // Fill in GPU ID number
  gpu.id = id;

  // Convert PCI bus ID to hex
  std::stringstream pcibus;
  pcibus << std::hex << gpu.prop.pciBusID;

  // Fill in PCI location
  pcibus >> gpu.pci;
  gpu.pci += ":" + std::to_string(gpu.prop.pciDomainID)
    + "." + std::to_string(gpu.prop.pciDeviceID);

  // Fill in GCN arch and hardware revision
  gpu.arch = gpu.prop.gcnArchName;
  gpu.rev = gpu.prop.asicRevision;

  // Identify codename and model
  int family_idx = -1;
  int gpuname_idx = -1;
  int ierr = identify_gpu(gpu.arch, family_idx, gpuname_idx, gputable);
  if (family_idx < 0) {    
    gpu.codename = "Unknown";
  } else {
    gpu.codename = gputable[family_idx].codename;
  } // family_idx
  if (gpuname_idx < 0) {
    gpu.name = "Unknown";
  } else {
    gpu.name = gputable[family_idx].gpuname.elem[gpuname_idx];
  } // gpuname_idx

  // Fill in HIP compute capability
  gpu.hip_cc = std::to_string(gpu.prop.major) + '.' + std::to_string(gpu.prop.minor);

  // Fill in whether GPU is a multi-GPU board
  gpu.is_multigpu = isNon0(gpu.prop.isMultiGpuBoard);

  // Fill in whether it's a APU or discrete GPU
  gpu.is_apu = isNon0(gpu.prop.integrated);

  // Fill in hardware execution parameters
  gpu.cuclk = gpu.prop.clockRate;
  gpu.instclk = gpu.prop.clockInstructionRate;
  gpu.cu_count = gpu.prop.multiProcessorCount;
  // gpu.thr_per_cu = gpu.prop.MaxThreadsPerMultiProcessor; // bug: always 0
  // gpu.reg_per_blk = gpu.prop.regsPerBlock; // bug: always 0

  // Fill in memory parameters
  gpu.memclk = gpu.prop.memoryClockRate;
  gpu.bussz = gpu.prop.memoryBusWidth;
  gpu.global_memsz = gpu.prop.totalGlobalMem;
  // gpu.l2_memsz = gpu.prop.l2CacheSize; // bug: always 0
  gpu.shared_memsz = gpu.prop.totalConstMem;
  gpu.shared_per_cu = gpu.prop.maxSharedMemoryPerMultiProcessor;
  gpu.shared_per_blk = gpu.prop.sharedMemPerBlock;
  gpu.pitch = gpu.prop.memPitch;
  gpu.ecc = isNon0(gpu.prop.ECCEnabled);
  gpu.large_bar = isNon0(gpu.prop.isLargeBar);
  gpu.managed_mem = isNon0(gpu.prop.managedMemory);
  gpu.managed_host = isNon0(gpu.prop.directManagedMemAccessFromHost);
  gpu.managed_concur = isNon0(gpu.prop.concurrentManagedAccess);
  gpu.maphost = isNon0(gpu.prop.canMapHostMemory);
  gpu.pageable = isNon0(gpu.prop.pageableMemoryAccess);
  gpu.pageable_host = isNon0(gpu.prop.pageableMemoryAccessUsesHostPageTables);

  // Fill in execution control flags
  gpu.concur = isNon0(gpu.prop.concurrentKernels);
  gpu.coop = isNon0(gpu.prop.cooperativeLaunch);
  gpu.coop_multi = isNon0(gpu.prop.cooperativeMultiDeviceLaunch);

  // Fill in compute parameters
  gpu.warpsz = gpu.prop.warpSize;
  gpu.thr_per_blk = gpu.prop.maxThreadsPerBlock;
  for (int i = 0; i < 3; i++) { gpu.max_thr[i] = gpu.prop.maxThreadsDim[i]; }
  for (int i = 0; i < 3; i++) { gpu.max_grid[i] = gpu.prop.maxGridSize[i]; }

  return 0;
}

// List all detected GPUs
int list_gpus (const int ngpus, const device_t* gpus) {
  for (int n = 0; n < ngpus; n++) {
    const device_t& gpu = gpus[n];
    std::cout << gpu.id << ". "
	      << gpu.pci
	      << " " << gpu.arch
	      << " " << gpu.name
	      << " (" << gpu.codename << ")" << std::endl;
  }
  return 0;
}

// Prompt and select a GPU id
int select_gpu (const int ngpus, int& selected_gpu, bool& all_gpus) {

  // Fill in default values
  selected_gpu = 0;
  all_gpus = false;

  // Show dialog
  bool valid = false;
  int input;
  while (!valid) {
    std::cout << "Select HIP device"
	      << " (" << ngpus << " = all devices):" << std::endl;
    std::cin >> input;

    // Sanitize input
    if (input <= ngpus) {
      valid = true;
      selected_gpu = input;
    } else {
      std::cout << "Invalid entry " << input << std::endl;
    }
  }

  if (selected_gpu == ngpus) {
    all_gpus = true;
  }
  return 0;
}

int print_gpu_info (const int id, const device_t gpu) {

  std::cout << "Device #" << gpu.id << ". "
	    << gpu.pci
	    << " " << gpu.arch
	    << " " << gpu.name
	    << " (" << gpu.codename << ")"
	    << " rev. " << gpu.rev << std::endl;

  std::cout << std::string(80, '-') << std::endl;

  std::cout << "HIP compute capability: " << gpu.hip_cc << std::endl;
  std::cout << "Device type: " << (gpu.is_apu ? "APU" : "Discrete GPU") << std::endl;
  std::cout << "Multi-GPU board: " << YesNo(gpu.is_multigpu) << std::endl;

  std::cout << std::string(80, '-') << std::endl;

  std::cout << "Number of compute units: " << gpu.cu_count << std::endl;
  std::cout << "Max clock frequency: " << (double)gpu.cuclk / (double)1000 << " MHz" << std::endl;
  std::cout << "Device-side instruction clock frequency: " << (double)gpu.instclk / (double)1000 << " MHz" << std::endl;

  std::cout << std::string(80, '-') << std::endl;

  std::cout << "Warp size: " << gpu.warpsz << std::endl;
  std::cout << "Max work items/threads per workgroup/threadblock: " << gpu.thr_per_blk << std::endl;
  std::cout << "Max dimensions per workgroup/threadblock: "
	    << gpu.max_thr[0] << " x " << gpu.max_thr[1] << " x " << gpu.max_thr[2] << std::endl;
  std::cout << "Max grid dimensions: "
	    << gpu.max_grid[0] << " x " << gpu.max_grid[1] << " x " << gpu.max_grid[2] << std::endl;

  std::cout << std::string(80, '-') << std::endl;
  
  std::cout << "Max memory clock frequency: " << gpu.memclk << " kHz" << std::endl;
  std::cout << "Memory bus width: " << gpu.bussz << " bits" << std::endl;
  std::cout << "Total global memory size: " << gpu.global_memsz << " bytes"
	    << " = " << toGiB(gpu.global_memsz) << " GiB" << std::endl;
  std::cout << "Total L1 / shared memory size: " << gpu.shared_memsz << " bytes"
	    << " = " << toMiB(gpu.shared_memsz) << " MiB" << std::endl;
  std::cout << "Maximum L1 / shared memory per CU: " <<  gpu.shared_per_cu << " bytes"
	    << " = " << toKiB(gpu.shared_per_cu) << " KiB" << std::endl;
  std::cout << "Maximum L1 / shared memory per threadblock: " <<  gpu.shared_per_blk << " bytes"
	    << " = " << toKiB(gpu.shared_per_blk) << " KiB" << std::endl;
  std::cout << "Maximum memory pitch: " << gpu.pitch << " bytes"
	    << " = " << toGiB(gpu.pitch) << " GiB" << std::endl;
  std::cout << "ECC memory: " << YesNo(gpu.ecc) << std::endl;
  std::cout << "Large PCI BAR enabled: " << YesNo(gpu.large_bar) << std::endl;
  std::cout << "Managed memory support: " << YesNo(gpu.managed_mem) << std::endl;
  std::cout << "Direct managed memory access from host: " << YesNo(gpu.managed_host) << std::endl;
  std::cout << "Concurrent managed memory access with CPU: " << YesNo(gpu.managed_concur) << std::endl;
  std::cout << "HIP can map host memory: " << YesNo(gpu.maphost) << std::endl;
  std::cout << "Coherent pageable memory access without hipHostRegister: " << YesNo(gpu.pageable) << std::endl;
  std::cout << "Pageable memory access via host's page tables: " << YesNo(gpu.pageable_host) << std::endl;

  std::cout << std::string(80, '-') << std::endl;

  std::cout << "Concurrent kernel execution: " << YesNo(gpu.concur) << std::endl;
  std::cout << "Cooperative launch support: " << YesNo(gpu.coop) << std::endl;
  std::cout << "Cooperative launch on multiple devices: " << YesNo(gpu.coop_multi) << std::endl;

  return 0;
}

#endif /* HELPERS_HIP_HPP */
