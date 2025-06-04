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

#include <iostream>
#include <string>
#include <vector>
#include <chrono>
#include <memory>
#include <dlfcn.h>
#include <filesystem>
#include <regex>
#include <future>

#include "connector/runner.h"
#include "connector/compiler.h"
#include "connector/libconnector.h"
#include "connector/domain.h"
#include "connector/buffer.h"
#include "connector/model.pb.h"
#include "connector/utils.h"

#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_executable.h"
#include "xla/pjrt/pjrt_stream_executor_client.h"
#include "xla/pjrt/cpu/cpu_client.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/log/initialize.h"
#include "xla/service/dump.h"
#include "tsl/platform/init_main.h"
#include "tsl/platform/logging.h"
#include "tsl/platform/env.h"
#include "tsl/platform/path.h"
#include "tsl/platform/protobuf.h"


namespace jcn {

    void Runner::initialize() {

        absl::InitializeLog();

        Logger logger = Logger::getlogger();

        char* env_var = std::getenv("JCN_PJRT_PATH");
        if (env_var == nullptr) {
            std::cerr << "Set JCN_PJRT_PATH to discover PJRT Plugins" << std::endl;
            return;
        }

        std::string plugin_path = std::string(std::getenv("JCN_PJRT_PATH"));

        try {
            // Infer a name
            std::regex pattern(R"(pjrt_plugin\.xla_(\w+)\.so)");

            for (const auto& entry : std::filesystem::directory_iterator(plugin_path)) {
                std::string path = entry.path().string();
                std::smatch match;

                if (!std::regex_search(path, match, pattern)) continue;

                absl::StatusOr<const PJRT_Api*> status_or_api = pjrt::LoadPjrtPlugin(match.str(1), path);

                if (status_or_api.ok()) {
                    logger.log(
                        LogLevel::INFO, "Loaded PJRT plugin " + match.str(1)
                    );
                } else {
                    std::cerr << "Failed to load PJRT plugin: " << status_or_api.status().ToString() << std::endl;
                }

            }
        } catch (const std::exception& e) {
            std::cerr << "Failed to load pjrt plugins: " << e.what() << std::endl;
        }

    }

    Runner::Runner(ConnectorConfig config, bool initialize) {

        if (initialize) {
            Runner::initialize();
        }

        // Singleton
        Logger logger = Logger::getlogger();

        absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client_or_status;

        if (config.backend == "cpu") {

          xla::CpuClientOptions create_options;
          create_options.asynchronous = true;

          client_or_status = xla::GetTfrtCpuClient(create_options);

        } else {

            absl::flat_hash_map<std::string, xla::PjRtValueType> create_options = {
                {"memory_fraction", static_cast<float>(config.memory_fraction)},
                {"visible_devices", std::vector<int64_t>({config.device})},
            };

            // Initialize the possible backends in the libconnector file
            absl::StatusOr<bool> status_or_success = pjrt::IsPjrtPluginInitialized(config.backend);
            if (!status_or_success.ok()) {
                throw std::runtime_error("Failed to initialize PjRtClient: " + status_or_success.status().ToString());
            }

            if (!status_or_success.value()) {
                absl::Status status = pjrt::InitializePjrtPlugin(config.backend);
                if (!status.ok()) {
                    throw std::runtime_error("Failed to initialize PjRtClient: " + status.ToString());
                }
            }

            // Get the client
            client_or_status = xla::GetCApiClient(config.backend, create_options);

        }

        if (!client_or_status.ok()) {
            throw std::runtime_error("Failed to initialize PjRtClient: " + client_or_status.status().ToString());
        }

        client = std::move(client_or_status).value();

        // Print devices
        absl::Span<xla::PjRtDevice* const> devices = client->devices();
        std::string device_list = "";
        for (int i = 0; i < devices.size(); i++) {
            device_list += std::string(devices[i]->ToString()) + ",";
	    }
	    logger.log(LogLevel::INFO, "Found devices [" + device_list + "]");

    }


    ModelProperties Runner::load_model(ModelConfig config) {
        // Singleton
        Logger logger = Logger::getlogger();

        newton = config.newton;

        atom_builder = std::make_unique<AtomBuilder>(
            config.atom_multiplier, config.newton);
        model = std::make_unique<chemsim::Model>();

        // Deserialize the protobuffer
        if (config.model.empty()) {
            throw std::runtime_error("Cannot load model: Model file is empty.");
        }

        if (!model->ParseFromString(config.model)) {
            throw std::runtime_error("Cannot load model: Model file is invalid or corrupted.");
        }

        // Pass the mlir module to the compiler
        compiler = std::make_unique<Compiler>(model->mlir_module());

        // Select from the available neighbor list types
        switch (model->neighbor_list().type()) {
            case chemsim::Model::SIMPLE_SPARSE:
                neighbor_list = std::make_unique<SimpleSparseNeighborList>();
                neighbor_list->initialize(config.neighbor_list_multipliers);

                logger.log(LogLevel::INFO, "Initialize SimpleSparseNeighborList");
                break;
            case chemsim::Model::SIMPLE_DENSE:
                neighbor_list = std::make_unique<SimpleDenseNeighborList>();
                neighbor_list->initialize(config.neighbor_list_multipliers);

                logger.log(LogLevel::INFO, "Initialize SimpleDenseNeighborList");
                break;
            case chemsim::Model::DEVICE_SPARSE:
                neighbor_list = std::make_unique<DeviceSparseNeighborList>();
                neighbor_list->initialize(config.neighbor_list_multipliers);

                logger.log(LogLevel::INFO, "Initialize DeviceSparseNeighborList");
                break;
        }

        return get_model_properties();

    }

    Results Runner::compute_forces(
        int lnum, int gnum, double **x, double **f, int *type, int inum,
        int *ilist, int *numneigh, int **firstneigh, bool list_changed,
        bool allow_recompile
    ) {

        // Singleton
        Logger logger = Logger::getlogger();

        int max_trials = 10;
        bool recompiled = false;

        for (int i = 0; i < max_trials; i++) {

            auto trial_start = std::chrono::high_resolution_clock::now();

            // First we build the domain and the neighbor list, then we can
            // determine the input shapes to the program

            AtomShapes atoms = atom_builder->get_shapes(lnum, gnum, allow_recompile);

            NeighborListShapes neighbors = neighbor_list->get_neighbor_list_shapes(
                atoms.n_atoms, inum, numneigh, allow_recompile);

            // Now we have all shapes setup to build the module if required.
            // If the module tried to recompile but failed due to disabled
            // recompilation, it will try again in the next call due to the
            // flag recompilation_required.
            // If recompilation is not necessary but allowed, it will depend
            // on how much the buffers are filled.
            recompilation_required |= !executable || atoms.reallocate || neighbors.reallocate;
            if (recompilation_required && allow_recompile) {
                recompiled |= true; // Track for statistics whether recompilation was necessary

                logger.log(LogLevel::INFO, "Recompilation necessary");

                compiler->compile(
                    atoms.n_atoms, neighbors.graph_shapes, neighbors.graph_types);

                absl::StatusOr<std::unique_ptr<xla::PjRtLoadedExecutable>> executable_or_status = client->Compile(
                    compiler->module(), compile_options);

                if (!executable_or_status.ok()) {
                    throw std::runtime_error("Failed to compile: " + executable_or_status.status().ToString());
                }

                executable = std::move(executable_or_status).value();

                // Print a cost analysis of the exectuable
                absl::StatusOr<absl::flat_hash_map<std::string, xla::PjRtValueType>> cost_analysis;
                cost_analysis = executable->GetCostAnalysis();
                if (cost_analysis.ok()) {
                    const absl::flat_hash_map<std::string, xla::PjRtValueType>& cost_map = cost_analysis.value();

                    auto it = cost_map.find("flops");
                    if (it != cost_map.end()) {
                        if (const float* flops = std::get_if<float>(&it->second)) {
                            flops_ = *flops;
                            logger.log(LogLevel::INFO,
                                "Cost analysis: " + std::to_string(*flops) + " flops."
                            );
                        } else {
                            std::cerr << "Error: 'flops' is not a float type" << std::endl;
                        }
                    } else {
                        std::cerr << "Error: 'flops' key not found in cost_map" << std::endl;
                    }
                } else {
                    std::cerr << "Failed to get cost analysis: " << cost_analysis.status().ToString() << std::endl;
                }

                recompilation_required = false; // Reset the recompilation flag

            } else if (recompilation_required) {
                throw jcn::RecompilationRequired(
                    "Recompilation required but not allowed. Please set allow_recompile to true.");
            }

            auto start = std::chrono::high_resolution_clock::now();

            // Only transfer new data to the GPU if necessary
            bool update = (recompiled || list_changed);

            // Now we have to create the buffers, i.e., copy the data onto
            // the device
            std::vector<xla::PjRtBuffer*> buffer_ptrs = atom_builder->build_domain(client.get(), config.device, lnum, gnum, x, type);

            // TODO: We have to add the gnum option to the neighbor list.
            //       This is only a workaround for the sparse neighbor list
            //       which includes the ghost atoms as senders.
            std::vector<xla::PjRtBuffer*> graph_buffers = neighbor_list->build_graph(
                client.get(), config.device, inum, ilist, numneigh, firstneigh, update);
            buffer_ptrs.insert(buffer_ptrs.end(), graph_buffers.begin(), graph_buffers.end());

            std::vector<std::vector<xla::PjRtBuffer*>> arg_handles = {buffer_ptrs};

            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> duration = end - start;

            logger.log(LogLevel::DEBUG, "Time taken for buffer creation: " + std::to_string(duration.count()) + " seconds");

            // Check if arg_handles is correctly populated
            if (arg_handles.empty() || arg_handles[0].empty()) {
                throw std::runtime_error("arg_handles is empty or not properly populated");
            }

            // No idea what to specify here...
            xla::ExecuteOptions execute_options;
            execute_options.untuple_result = true;
            //  execute_options.execution_mode = xla::ExecuteOptions::ExecutionMode::kSynchronous;


                start = std::chrono::high_resolution_clock::now();


            // Use std::async to execute the function asynchronously
            std::future<absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>>> future_results =
                std::async(std::launch::async, [&]() {
                    return executable->Execute(
                        absl::Span<const std::vector<xla::PjRtBuffer*>>(arg_handles),
                        execute_options
                    );
                });

            // Wait for the results to be ready
            absl::StatusOr<std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>>> results = future_results.get();

            if (!results.ok()) {
                throw std::runtime_error("Failed to execute: " + results.status().ToString());
            }

            // Now we have to copy the results back to the host
            std::vector<std::vector<std::unique_ptr<xla::PjRtBuffer>>> results_buffers = std::move(results).value();

            bool success = neighbor_list->evaluate_statistics(results_buffers, allow_recompile);

            end = std::chrono::high_resolution_clock::now();
            duration = end - start;

            logger.log(LogLevel::DEBUG, "Time taken for computation: " + std::to_string(duration.count()) + " seconds");

            // Write back the results
            double potential = atom_builder->evaluate_domain(
                success, lnum, gnum, f, results_buffers);

            auto trial_end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> trial_duration = trial_end - trial_start;

            logger.log(LogLevel::DEBUG, "Time taken for trial: " + std::to_string(trial_duration.count()) + " seconds");

            results_buffers.clear();

            Results compute_results;
            compute_results.potential = potential;
            compute_results.stats.flops = flops_;
            compute_results.stats.recompiled = recompiled;

            // Finished
            if (success) return compute_results;

        }

        throw std::runtime_error("Failed to compute forces after " + std::to_string(max_trials) + " trials");

    }

    ModelProperties Runner::get_model_properties() {
        // Singleton
        Logger logger = Logger::getlogger();

        ModelProperties properties;

        if (!model) {
            throw std::runtime_error("Model is not initialized");
        }

        // Sufficient number of ghost atomus must be communicated.
        // The communication cutoff depends on the number of message passing
        // steps which effectively increase the cutoff distance.
        int multiplier;
        if (newton) {
            multiplier = model->neighbor_list().nbr_order()[0];
        } else {
            multiplier = model->neighbor_list().nbr_order()[1];
        }
        properties.comm_dist = multiplier * model->neighbor_list().cutoff();

        if (model->has_unit_style()) {
            properties.unit_style = model->unit_style().c_str();
        } else {
            properties.unit_style = "real"; // Define this as default
        }

        switch (model->neighbor_list().type()) {
            case chemsim::Model::SIMPLE_SPARSE:
            case chemsim::Model::SIMPLE_DENSE:
                // Neighbor list cutoff must be larger than the model cutoff
                properties.cutoff = model->neighbor_list().cutoff();

                if (multiplier > 1) {
                    // Ghost atoms only required if more than the next neighbor
                    // can affect the local energy of a particle
                    properties.neighbor_list.include_ghosts = true;
                    logger.log(LogLevel::INFO,
                        "Include ghosts: " + std::to_string(properties.neighbor_list.include_ghosts)
                    );
                };
                if (model->neighbor_list().has_half_list()) {
                    properties.neighbor_list.half_list = model->neighbor_list().half_list();
                    logger.log(LogLevel::INFO,
                        "Use half list only " + std::to_string(properties.neighbor_list.half_list)
                    );
                };

                break;
            case chemsim::Model::DEVICE_SPARSE:
                // Does not specify a cutoff for the particles as neighbor
                // list is computed on the device
                properties.cutoff = 0.0;

                break;
        }

        logger.log(LogLevel::INFO,
            std::string("Model properties:") +
            "\n\t-Cutoff: " + std::to_string(properties.cutoff) +
            "\n\t-Com. distance: " + std::to_string(properties.comm_dist) +
            "\n\t-Unit style: " + properties.unit_style
        );

        return properties;
    }

} // namespace jcn
