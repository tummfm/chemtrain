/*
Copyright 2025 Multiscale Modeling of Fluid Materials, TU Munich

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

#include "libconnector.h"
#include "runner.h"

#include <iostream>
#include <string>
#include <vector>
#include <memory>  // For std::unique_ptr
#include <dlfcn.h>

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/service/dump.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"


namespace jcn {

    bool Connector::initialized = false;

    Connector::~Connector() = default;

    class Connector::Impl {
    public:
        Impl(ConnectorConfig config, bool initialize)
            : runner(config, initialize) {};
        ~Impl() = default;

        Runner runner;

    };

    Results Connector::compute_force(int lnum, int gnum, double **x, double** f,
        int *type, int inum, int *ilist, int *numneigh, int **firstneigh,
        bool list_changed, bool allow_recompile) {

        return impl_->runner.compute_forces(
          lnum, gnum, x, f, type, inum, ilist, numneigh, firstneigh,
          list_changed, allow_recompile);
    }

    Connector::Connector(ConnectorConfig config) {
        std::cout << "Connector constructor" << std::endl;

        impl_ = std::make_unique<Impl>(config, !initialized);

        if (!initialized) {
            initialized = true;
        }

    };

    ModelProperties Connector::load_model(ModelConfig config) {
        return impl_->runner.load_model(config);
    }


} // namespace jcn