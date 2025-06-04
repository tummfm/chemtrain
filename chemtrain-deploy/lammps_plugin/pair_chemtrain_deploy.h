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

#ifndef LMP_PAIR_MORSE2_H
#define LMP_PAIR_MORSE2_H

#include "pair.h"
#include "libconnector.h"

namespace LAMMPS_NS {

class ChemtrainDeploy : public Pair {
 public:
  ChemtrainDeploy(class LAMMPS *);
  ~ChemtrainDeploy() override;
  void compute(int, int) override;

  void settings(int, char **) override;
  void coeff(int, char **) override;
  void init_style() override;
  double init_one(int, int) override;
  void finish() override;

 protected:
//  bool allocated;

  // double cut_global;
  double **cut;
  double **xold;

  // Statistics
  int recompilations;
  double flops;

  jcn::ModelProperties model_properties;

  bool check_distance();

  std::unique_ptr<jcn::Connector> connector;

  virtual void allocate();
};

}    // namespace LAMMPS_NS

#endif

