/* ----------------------------------------------------------------------------
    chemtrain-deploy - LAMMPS plugin
    Copyright (C) 2025  Multiscale Modeling of Fluid Materials, TU Munich

    This program is free software; you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation; either version 2 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    See the LICENSE file in the directory of this file.
---------------------------------------------------------------------------- */
#include "pair_chemtrain_deploy.h"

#include "libconnector.h"

#include "atom.h"
#include "comm.h"
#include "error.h"
#include "force.h"
#include "memory.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "update.h"
#include "utils.h"

#include <cmath>
#include <cstring>
#include <iostream>
#include <fstream>
#include <chrono>
#include <stdlib.h>
#include <sstream>

using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

ChemtrainDeploy::ChemtrainDeploy(LAMMPS *lmp) : Pair(lmp)
{
  writedata = 0;
  single_enable = 0;
  restartinfo = 0;
  one_coeff = 1;
  manybody_flag = 1;
}

/* ---------------------------------------------------------------------- */

ChemtrainDeploy::~ChemtrainDeploy()
{
  if (allocated) {

  memory->destroy(setflag);
  memory->destroy(cutsq);
  memory->destroy(cut);
  memory->destroy(xold);

  }
}

bool ChemtrainDeploy::check_distance() {
  double **x = atom->x;
  int nlocal = atom->nlocal;

  double deltasq = 2.0 * 2.0; // Hard coded skin distance

  int flag = 0;
  for (int i = 0; i < nlocal; i++) {
    double delx = x[i][0] - xold[i][0];
    double dely = x[i][1] - xold[i][1];
    double delz = x[i][2] - xold[i][2];
    double rsq = delx * delx + dely * dely + delz * delz;
    if (rsq > deltasq) {
      flag = 1;
      break;
    }
  }

  int flagall;
  MPI_Allreduce(&flag, &flagall, 1, MPI_INT, MPI_MAX, world);

  bool update_list = (flagall > 0);

  if (update_list) {
    for (int i = 0; i < atom->nlocal; i++) {
      std::memcpy(xold[i], atom->x[i], 3 * sizeof(double));
    }
  }

  return update_list;

}


void ChemtrainDeploy::compute(int eflag, int vflag)
{

  ev_init(eflag, vflag);

  auto start = std::chrono::high_resolution_clock::now();

  // Check if neighborlist was updated just in this timestep resulting in newly communicated atoms
  bool update_list = check_distance() || (neighbor->ago == 0);

  // Number of sender atoms can change depending on the ghost setting of the
  // neighbor list
  int inum = list->inum + list->gnum;

  // Will try to run the step without recompilation. If recompilation is
  // required on one device, all other devices will check whether a
  // recompilation is required soon.

  int retry_flag = 0;
  jcn::Results results;

  try {
    results = connector->compute_force(
      atom->nlocal, atom->nghost, atom->x, atom->f, atom->type,
      inum, list->ilist, list->numneigh, list->firstneigh, update_list, false
    );
  } catch (const jcn::RecompilationRequired& e) {
    retry_flag = 1;
  }

  int retry_flag_all;
  MPI_Allreduce(&retry_flag, &retry_flag_all, 1, MPI_INT, MPI_MAX, world);

  // If one device must recompile, give all other devices the possibility to
  // recompile as well if the buffers are close to beeing full
  if (retry_flag_all > 0) {
    results = connector->compute_force(
      atom->nlocal, atom->nghost, atom->x, atom->f, atom->type,
      inum, list->ilist, list->numneigh, list->firstneigh, update_list, true
    );
  }

  // Pass the evaluated potential energy to LAMMPS
  if (eflag) {
    eng_vdwl = results.potential;
  }

  // TODO: Log additional statistics

  // Save statistics
  flops += static_cast<double>(results.stats.flops);
  recompilations += static_cast<int>(results.stats.recompiled);

  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;

  if (vflag_fdotr) virial_fdotr_compute();

}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void ChemtrainDeploy::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag, n + 1, n + 1, "pair:setflag");
  memory->create(cutsq, n + 1, n + 1, "pair:cutsq");
  memory->create(cut, n + 1, n + 1, "pair:cut");
  memory->create(xold, atom->nmax, 3, "pair:xold");

  for (int i = 1; i <= n; i++) {
      for (int j = 1; j <= n; j++) setflag[i][j] = 0;
  }
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void ChemtrainDeploy::settings(int narg, char **arg)
{
  // Settings then the pair_style command is called
  if (narg < 1) error->all(FLERR, "Illegal jax_connect command");

  jcn::ConnectorConfig config;

  // Assigns the devices based on the local rank exclusively to one MPI process
  int device_id = 0;
  char* local_rank;

  if ((local_rank = getenv("SLURM_LOCALID"))) {
    device_id = std::stoi(local_rank);
    utils::logmesg(lmp, "Assign device based on SLURM_LOCALID");
  }
  if ((local_rank = getenv("OMPI_COMM_WORLD_LOCAL_RANK"))) {
    utils::logmesg(lmp, "Assign device based on OMPI_COMM_WORLD_LOCAL_RANK");
    device_id = std::stoi(local_rank);
  }
  if ((local_rank = getenv("MV2_COMM_WORLD_LOCAL_RANK"))) {
    utils::logmesg(lmp, "Assign device based on MV2_COMM_WORLD_LOCAL_RANK");
    device_id = std::stoi(local_rank);
  }
  if ((local_rank = getenv("FLUX_TASK_LOCAL_ID"))) {
    utils::logmesg(lmp, "Assign device based on FLUX_TASK_LOCAL_ID");
    device_id = std::stoi(local_rank);
  }
  if ((local_rank = getenv("PMI_LOCAL_RANK"))) {
    utils::logmesg(lmp, "Assign device based on PMI_LOCAL_RANK");
    device_id = std::stoi(local_rank);
  }

  config.backend = std::string(arg[0]);
  config.device = device_id;

  // Optional argument to increase the memory fraction
  if (narg > 1) {
    config.memory_fraction = std::stof(arg[1]);
  }


  // Initialize the model within XLA
  connector = std::make_unique<jcn::Connector>(config);

}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void ChemtrainDeploy::coeff(int narg, char **arg)
{

	if (!allocated) allocate();

    if (narg < 4) error->all(FLERR, "Illegal jax_connect command");

    std::string exported_model_path = arg[2];

    const float atom_multiplier = std::stof(arg[3]);

    std::vector<float> neighbor_list_multipliers;
    for (int i = 4; i < narg; i++) {
        neighbor_list_multipliers.push_back(std::stof(arg[i]));
    }

    // Load the exported model
    std::ifstream file(exported_model_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + exported_model_path);
    }

    std::string exported_model((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());

    jcn::ModelConfig config;

    config.model = exported_model;
    config.neighbor_list_multipliers = neighbor_list_multipliers;
    config.atom_multiplier = atom_multiplier;
    config.newton = force->newton_pair;

    // Set the flags to mark initialization of all pair coefficients
    int ilo, ihi, jlo, jhi;
    utils::bounds(FLERR, arg[0], 1, atom->ntypes, ilo, ihi, error);
    utils::bounds(FLERR, arg[1], 1, atom->ntypes, jlo, jhi, error);
    for (int i = ilo; i <= ihi; i++) {
      for (int j = MAX(jlo, i); j <= jhi; j++) {
        setflag[i][j] = 1;
      }
    }

    // Load the model
    model_properties = connector->load_model(config);

    // Check whether correct units are defined
    std::string req_style = update->unit_style;
    std::string set_style = model_properties.unit_style;
    if (set_style != req_style) {
        error->all(FLERR,
            "The units of the model do not match the unit style {:s}. "
            "Please use the units from {:s} to {:s}.",
            req_style, set_style
        );
    }

}


void ChemtrainDeploy::init_style()
{

  // Reset the statistics
  recompilations = 0;
  flops = 0;

  // Full list not required as we can simply reverse all undirected edges
  int request = NeighConst::REQ_DEFAULT;

  if (model_properties.neighbor_list.include_ghosts) {
    request |= NeighConst::REQ_GHOST;
  }

  if (!model_properties.neighbor_list.half_list || force->newton) {
    // It seems like setting newton to true requires a full list
    request |= NeighConst::REQ_FULL;
  }

  neighbor->add_request(this, request);

}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double ChemtrainDeploy::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR, "Not all pair coeffs are set");

  // Initialize the old atom positions
  for (int i = 0; i < atom->nlocal; i++) {
      std::memcpy(xold[i], atom->x[i], 3 * sizeof(double));
  }

  // Check whether communication cutoff is large enough
  double min_comm_dist = model_properties.comm_dist + neighbor->skin;
  if (min_comm_dist > comm->get_comm_cutoff()) {
    error->all(
      FLERR, "Communication cutoff is too small for the model. Increase "
      "the communication cutoff to at least {:.4f}.", min_comm_dist
    );
  }

  return model_properties.cutoff;
}

void ChemtrainDeploy::finish()
{
  // Log the statistics
  int min_comp, max_comp, sum_comp, num_procs;
  double min_flops, max_flops, sum_flops;

  MPI_Allreduce(&recompilations, &min_comp, 1, MPI_INT, MPI_MIN, world);
  MPI_Allreduce(&recompilations, &max_comp, 1, MPI_INT, MPI_MAX, world);
  MPI_Allreduce(&recompilations, &sum_comp, 1, MPI_INT, MPI_SUM, world);
  MPI_Allreduce(&flops, &min_flops, 1, MPI_DOUBLE, MPI_MIN, world);
  MPI_Allreduce(&flops, &max_flops, 1, MPI_DOUBLE, MPI_MAX, world);
  MPI_Allreduce(&flops, &sum_flops, 1, MPI_DOUBLE, MPI_SUM, world);
  MPI_Comm_size(world, &num_procs);

  double avg_comp = static_cast<double>(sum_comp) / static_cast<double>(num_procs);
  double avg_flops = sum_flops / static_cast<double>(num_procs);

  utils::logmesg(
      lmp, "\n==== JaxConnect Summary =========.\n"
           "- Recompilations: {:d} min / {:.2f} avg / {:d} max. / {:d} total \n"
           "- Estimated FLOP: {:.2e} min / {:.2e} avg / {:.2e} max. / {:.2e} total\n\n",
            min_comp, avg_comp, max_comp, sum_comp,
            min_flops, avg_flops, max_flops, sum_flops
           );
}
