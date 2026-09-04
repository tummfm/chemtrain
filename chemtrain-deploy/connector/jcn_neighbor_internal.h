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

#ifndef CHEMTRAIN_DEPLOY_CONNECTOR_JCN_NEIGHBOR_INTERNAL_H_
#define CHEMTRAIN_DEPLOY_CONNECTOR_JCN_NEIGHBOR_INTERNAL_H_

#include "connector/jcn_api.h"

struct jcn_neighbor_list {
  jcn_neighbor_format format = JCN_NEIGHBOR_UNSPECIFIED;
  jcn_sparse_neighbors sparse{};
  jcn_dense_neighbors dense{};
};

#endif  // CHEMTRAIN_DEPLOY_CONNECTOR_JCN_NEIGHBOR_INTERNAL_H_
