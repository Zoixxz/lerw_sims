import numpy as np
from typing import List
import math

import random
from scipy.stats import zipf


class MHS:
    def __init__(self, alpha: float, d: int):
        """
        alpha: decay parameter (goal: have look at alpha < 2)
        d: dimension of Z^d
        """
        if alpha <= 0:
            raise ValueError("Alpha has to be positive")
        self.alpha = alpha
        self.d = d

    def pdf(self, x: tuple[int, ...]) -> float:
        """
        norm in 3d with param alpha
        ! we do not need the correct normalization in metro hastings !
        """
        norm = math.sqrt(sum(x_i ** 2 for x_i in x))
        if not norm < 1e-5:  # FIXME? these should all be integers, so norm == 0 ?
            return 1 / (norm) ** (self.d + self.alpha)
        return 0

    def proposal(self, current: tuple[int, ...]) -> tuple[int, ...]:
        """
        generate porposal step form zip distro in each coordinate
        """
        # TODO: pass sampling function as argument Q: which one is the best?
        # TODO: play around with parameter "alpha / d" works well
        # TODO: criterion for mixing?
        # TODO: can we use pareto/continuous distributions for c++?
        directions = [random.choice([1, -1]) for _ in range(self.d)]
        coord_steps = [
            zipf(1 + (self.alpha / self.d)).rvs() for _ in range(self.d)
        ]  # choose s < alpha for better exploration
        # bad proposal
        # coord_steps = [random.choice([1, -1]) for _ in range(self.d)]

        # alternative proposal setep using uniform RV on circle and Pareto RV
        # point = np.zeros(self.d)
        # while np.linalg.norm(point) == 0:  # handle div by 0 case
        #    point = np.random.normal(loc=0, scale=1, size=self.d)
        # directions = point/np.linalg.norm(point)
        # u = np.random.rand()
        # while u == 0:
        #    u = np.random.rand()
        #
        # r = u ** (-1/self.alpha)  # this is a pareto distribution
        # delta = list(np.round(r * point).astype(int))

        delta = tuple(dir * coord for dir, coord in zip(directions, coord_steps))

        return tuple(c + d for c, d in zip(current, delta))

    def acceptance_proba(
        self, current_point: tuple[int, ...], proposal_point: tuple[int, ...]
    ) -> float:
        """
        compute Metro Hastings acceptance
        """
        p_proposal: float = self.pdf(proposal_point)
        p_current: float = self.pdf(current_point)

        if p_current == 0:
            return 1 if p_proposal > 0 else 0
        return min(1, p_proposal / p_current)

    def sample_gen(self, N: int, discard_N: int = int(1e3)) -> List[tuple[int, ...]]:
        """
        walk_ended: bool
        N: number of samples to generate
        discard_N: number of samples to discard (mixing of chain)

        returns: list containing all the samples (length N with discard_N samples discarded)
        """
        # TODO: online generation
        # TODO: modify to adapt to take current_point as parameter
        current_point: tuple[int, ...] = (0,) * self.d  # initalize start at 0
        samples: List[tuple[int, ...]] = []

        # meta info for eval
        accepted: int = 0  # accepted steps
        total: int = 1  # total setps

        total_iterations = N + discard_N

        for i in range(total_iterations):
            total += 1
            proposed_point = self.proposal(current_point)
            alpha = self.acceptance_proba(current_point, proposed_point)

            if np.random.random() < alpha:
                current_point = proposed_point
                accepted += 1

            if i >= discard_N:
                samples.append(current_point)

            if i % int(discard_N * 0.1) == 0 and i <= discard_N:
                acceptance_rate = accepted / total
                print(f"acc. rate during discarding phase: {acceptance_rate:.1f}")
        return samples
