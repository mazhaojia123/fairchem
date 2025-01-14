from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import TYPE_CHECKING

import ase
import torch
from torch_scatter import scatter
from .linesearch_torch import LineSearch  # Import LineSearch

import time

if TYPE_CHECKING:
    from .optimizable import OptimizableBatch

# Removed LineSearch class

class BFGSLineSearch:
    """
    Port of BFGSLineSearch from bfgslinesearch.py, adapted to PyTorch
    and batched operations, mirroring lbfgs_torch.py structure.
    """
    def __init__(
        self,
        optimizable_batch,
        maxstep=0.2,
        alpha=10.0,
        c1=0.23,
        c2=0.46,
        stpmax=50.0,
        device='cpu'
    ):
        # ...existing code...
        self.optimizable = optimizable_batch
        self.maxstep = maxstep
        self.alpha = alpha
        self.c1 = c1
        self.c2 = c2
        self.stpmax = stpmax
        # self.linesearch = LineSearch()
        self.H = None
        self.r0 = None
        self.g0 = None
        self.e0 = None
        self.p = None
        self.alpha_k = None
        self.no_update = False
        self.replay = False
        self.nsteps = 0
        self.device = device
        self.force_calls = 0
        # ...other init
    
    def step(self):
        optimizable = self.optimizable
        forces = optimizable.get_forces().to(self.device)
        # print(f'&&& forces: {forces}')
        r = optimizable.get_positions().reshape(-1).to(self.device)
        # print(f'&&& r: {r}')
        g = -forces.reshape(-1) / self.alpha
        p0 = self.p
        # print(f'&&& r: {r}')
        # print(f'&&& g: {g}')
        # print(f'&&& self.r0: {self.r0}')
        # print(f'&&& self.g0: {self.g0}')
        # print(f'&&& p0: {p0}')
        self.update(r, g, self.r0, self.g0, p0)
        e = self.func(r)

        self.p = -torch.matmul(self.H, g)
        p_size = torch.sqrt((self.p**2).sum())
        if p_size <= torch.sqrt(torch.tensor(len(optimizable) * 1e-10, device=self.device)):
            self.p /= (p_size / torch.sqrt(torch.tensor(len(optimizable) * 1e-10, device=self.device)))
        ls = LineSearch()
        # print(f'&&& self.p: {self.p}')
        # print(f'&&& g: {g}')
        # print(f'&&& e: {e}')
        # print(f'&&& e0: {self.e0}')
        # print(f'&&& self.maxstep: {self.maxstep}')
        # print(f'&&& self.c1: {self.c1}')
        # print(f'&&& self.c2: {self.c2}')
        # print(f'&&& self.stpmax: {self.stpmax}')

        self.alpha_k, e, self.e0, self.no_update = \
            ls._line_search(self.func, self.fprime, r, self.p, g, e, self.e0,
                            maxstep=self.maxstep, c1=self.c1,
                            c2=self.c2, stpmax=self.stpmax, cur_step=self.nsteps)
        # print(f'&&& self.alpha_k: {self.alpha_k}')
        # print(f'&&& e: {e}')
        # print(f'&&& e0: {self.e0}')
        # print(f'&&& no_update: {self.no_update}')
        if self.alpha_k is None:
            raise RuntimeError("LineSearch failed!")

        dr = self.alpha_k * self.p
        optimizable.set_positions((r + dr).reshape(len(optimizable), -1))
        self.r0 = r
        self.g0 = g

    def update(self, r, g, r0, g0, p0):
        # self.I = torch.eye(len(self.optimizable) * 3, dtype=torch.float32, device=self.device)
        self.I = torch.eye(len(self.optimizable) * 3, dtype=torch.int64, device=self.device)
        if self.H is None:
            self.H = torch.eye(3 * len(self.optimizable), device=self.device)
            return
        else:
            dr = r - r0
            dg = g - g0
            if not (((self.alpha_k or 0) > 0 and
                    torch.abs(torch.dot(g, p0)) - torch.abs(torch.dot(g0, p0)) < 0) or
                    self.replay):
                return
            if self.no_update is True:
                print('skip update')
                return

            try:
                rhok = 1.0 / (torch.dot(dg, dr))
            except ZeroDivisionError:
                rhok = 1000.0
                print("Divide-by-zero encountered: rhok assumed large")
            if torch.isinf(rhok):
                rhok = 1000.0
                print("Divide-by-zero encountered: rhok assumed large")
            A1 = self.I - dr[:, None] * dg[None, :] * rhok
            A2 = self.I - dg[:, None] * dr[None, :] * rhok
            self.H = (torch.matmul(A1, torch.matmul(self.H, A2)) +
                      rhok * dr[:, None] * dr[None, :])

    def func(self, x):
        self.optimizable.set_positions(x.reshape(-1, 3))
        return self.optimizable.get_potential_energy() / self.alpha

    def fprime(self, x):
        self.optimizable.set_positions(x.reshape(-1, 3))
        self.force_calls += 1
        forces = self.optimizable.get_forces().reshape(-1)
        return - forces / self.alpha

    def replay_trajectory(self, traj):
        self.replay = True
        from ase.utils import IOContext

        with IOContext() as files:
            if isinstance(traj, str):
                from ase.io.trajectory import Trajectory
                traj = files.closelater(Trajectory(traj, mode='r'))

            r0 = None
            g0 = None
            for i in range(len(traj) - 1):
                r = traj[i].get_positions().ravel()
                g = - traj[i].get_forces().ravel() / self.alpha
                self.update(r, g, r0, g0, self.p)
                self.p = -torch.matmul(self.H, g)
                r0 = r.copy()
                g0 = g.copy()
            self.r0 = r0
            self.g0 = g0

    def log(self, forces=None):
        if forces is None:
            forces = self.optimizable.get_forces()
        fmax = torch.sqrt((forces**2).sum(axis=1).max())
        # print(f'&&& forces in log: {forces}')
        # print(f'&&& fmax in log: {fmax}')
        e = self.optimizable.get_potential_energy()
        T = time.localtime()
        name = self.__class__.__name__
        if self.nsteps == 0:
            print('%s  %4s[%3s] %8s %15s  %12s' %
                  (' ' * len(name), 'Step', 'FC', 'Time', 'Energy', 'fmax'))
        print('%s:  %3d[%3d] %02d:%02d:%02d %15.6f %12.4f' %
              (name, self.nsteps, self.force_calls, T[3], T[4], T[5], e, fmax))

    def run(self, fmax, steps):
        self.fmax = fmax
        self.steps = steps

        forces = self.optimizable.get_forces()
        self.log(forces=forces)

        while self.nsteps < steps and not self.optimizable.converged(
            forces=forces, fmax=self.fmax
        ):
            self.step()
            self.nsteps += 1
            forces = self.optimizable.get_forces()
            self.log(forces=forces)
            
        
        return self.optimizable.converged(forces=forces, fmax=self.fmax)

# ...existing code...