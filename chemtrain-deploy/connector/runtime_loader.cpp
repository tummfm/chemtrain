/*
Copyright 2026 Multiscale Modeling of Fluid Materials, TU Munich

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "connector/runtime_loader.h"

#include <dlfcn.h>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

namespace jcn {
namespace {

const int kConnectorLibraryAnchor = 0;

using RegisterFfiProvider = int (*)(const XLA_FFI_Api*, const char*);
using GetXlaFfiApi = const XLA_FFI_Api* (*)();

std::vector<void*>& FfiProviderHandles() {
  static auto* handles = new std::vector<void*>;
  return *handles;
}

std::filesystem::path ConnectorLibraryDirectory() {
  Dl_info library_info{};
  if (dladdr(&kConnectorLibraryAnchor, &library_info) == 0 ||
      library_info.dli_fname == nullptr) {
    throw std::runtime_error("Failed to locate libconnector.so");
  }

  return std::filesystem::canonical(library_info.dli_fname).parent_path();
}

std::vector<std::string> SplitSearchPath(const char* raw_path) {
  std::vector<std::string> paths;
  std::string value(raw_path);
  std::size_t begin = 0;
  while (begin <= value.size()) {
    const std::size_t end = value.find(':', begin);
    std::string path = value.substr(begin, end - begin);
    if (path.empty()) {
      throw std::runtime_error("JCN_FFI_PATH contains an empty directory");
    }
    paths.push_back(std::move(path));
    if (end == std::string::npos) break;
    begin = end + 1;
  }
  return paths;
}

bool IsFfiProvider(const std::filesystem::path& path) {
  const std::string name = path.filename().string();
  return name.rfind("libjcn_ffi_", 0) == 0 && path.extension() == ".so";
}

}  // namespace

std::string PjrtDirectory() {
  const char* configured = std::getenv("JCN_PJRT_PATH");
  if (configured == nullptr) {
    return (ConnectorLibraryDirectory() / "pjrt").string();
  }
  if (configured[0] == '\0') {
    throw std::runtime_error("JCN_PJRT_PATH is empty");
  }
  if (std::string(configured).find(':') != std::string::npos) {
    throw std::runtime_error("JCN_PJRT_PATH must contain one directory");
  }
  return configured;
}

std::vector<std::string> FfiProviderDirectories() {
  const char* configured = std::getenv("JCN_FFI_PATH");
  if (configured == nullptr) {
    return {(ConnectorLibraryDirectory() / "ffi").string()};
  }
  if (configured[0] == '\0') return {};
  return SplitSearchPath(configured);
}

const char* XlaFfiPlatformForBackend(const std::string& backend) {
  if (backend == "cuda") return "CUDA";
  if (backend == "cpu") return "Host";
  return nullptr;
}

const XLA_FFI_Api* GetPjrtFfiApi(const std::string& pjrt_plugin_path) {
  void* plugin = dlopen(pjrt_plugin_path.c_str(), RTLD_NOW | RTLD_NOLOAD);
  if (plugin == nullptr) {
    const char* message = dlerror();
    throw std::runtime_error("Loaded PJRT plugin has no XLA FFI API anchor " +
                             pjrt_plugin_path + ": " +
                             (message == nullptr ? "unknown error" : message));
  }

  dlerror();
  GetXlaFfiApi get_ffi_api = nullptr;
  *reinterpret_cast<void**>(&get_ffi_api) =
      dlsym(plugin, "XLA_FFI_GetPluginApi");
  const char* symbol_error = dlerror();
  if (symbol_error != nullptr || get_ffi_api == nullptr) {
    dlclose(plugin);
    throw std::runtime_error("PJRT plugin " + pjrt_plugin_path +
                             " does not export XLA_FFI_GetPluginApi");
  }

  const XLA_FFI_Api* api = get_ffi_api();
  dlclose(plugin);
  if (api == nullptr || api->struct_size < XLA_FFI_Api_STRUCT_SIZE ||
      api->api_version.struct_size < XLA_FFI_Api_Version_STRUCT_SIZE ||
      api->api_version.major_version != XLA_FFI_API_MAJOR ||
      api->XLA_FFI_Handler_Register == nullptr) {
    throw std::runtime_error("PJRT plugin " + pjrt_plugin_path +
                             " returned an incomplete XLA FFI API");
  }
  return api;
}

void RegisterFfiProviders(const XLA_FFI_Api* api, const std::string& backend,
                          const std::vector<std::string>& search_directories) {
  const char* xla_platform = XlaFfiPlatformForBackend(backend);
  if (api == nullptr || xla_platform == nullptr) {
    throw std::runtime_error("FFI registration requires a supported backend");
  }

  std::set<std::filesystem::path> loaded_paths;
  for (const std::string& search_directory : search_directories) {
    const std::filesystem::path backend_directory =
        std::filesystem::path(search_directory) / backend;
    if (!std::filesystem::exists(backend_directory)) continue;
    if (!std::filesystem::is_directory(backend_directory)) {
      throw std::runtime_error("Invalid FFI provider directory: " +
                               backend_directory.string());
    }

    std::vector<std::filesystem::path> providers;
    for (const auto& entry :
         std::filesystem::directory_iterator(backend_directory)) {
      if (entry.is_regular_file() && IsFfiProvider(entry.path())) {
        providers.push_back(entry.path());
      }
    }
    std::sort(providers.begin(), providers.end());

    for (const std::filesystem::path& provider : providers) {
      const std::filesystem::path canonical_provider =
          std::filesystem::canonical(provider);
      if (!loaded_paths.insert(canonical_provider).second) continue;

      void* handle = dlopen(canonical_provider.c_str(), RTLD_NOW | RTLD_LOCAL);
      if (handle == nullptr) {
        const char* message = dlerror();
        throw std::runtime_error("Failed to load FFI provider " +
                                 canonical_provider.string() + ": " +
                                 (message == nullptr ? "unknown error" : message));
      }

      dlerror();
      RegisterFfiProvider register_ffi = nullptr;
      *reinterpret_cast<void**>(&register_ffi) = dlsym(handle, "RegisterFFi");
      const char* symbol_error = dlerror();
      if (symbol_error != nullptr || register_ffi == nullptr) {
        dlclose(handle);
        throw std::runtime_error("FFI provider " + canonical_provider.string() +
                                 " does not export RegisterFFi");
      }

      // Retain the DSO before registration because successful handlers cannot
      // be unregistered if a later handler in the same provider fails.
      FfiProviderHandles().push_back(handle);
      if (register_ffi(api, xla_platform) != 0) {
        throw std::runtime_error("RegisterFFi failed for " +
                                 canonical_provider.string() + " on " +
                                 xla_platform);
      }

      std::cout << "[JCN] Registered FFI provider " << canonical_provider
                << " for " << backend << std::endl;
    }
  }
}

}  // namespace jcn
