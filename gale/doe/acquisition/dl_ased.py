"""
GALe - Global Adaptive Learning
@author: Sven Lämmle

DoE - Acquisition
"""
from copy import deepcopy
from itertools import permutations
from typing import List, Optional

import numpy as np
from joblib import Parallel, delayed
from scipy.spatial.qhull import ConvexHull, Delaunay

from gale.doe.utils import edge_sampling
from gale.models import SurrogateRegressionModel, cook_model

from .adaptive_sampling import AdaptiveSampling


class DLASED(AdaptiveSampling):
    """
    Discrepancy criterion and Leave one out error-based Adaptive Sequential Experiment
    Design (DL-ASED)

    Literature
    ----------
    FANG Ke, Zhou Yuchen, Ma Ping - An adaptive sequential experiment design method for
    model validation. Chinese Journal of Aeronautics 33.6, 2019.
    """

    name = "Discrepancy Crit. and LOO Error based Adaptive Seq. Exp. Design"
    short_name = "DL-ASED"
    optimizer_used = False

    def __init__(
        self, bounds, n_init=None, rnd_state=None, verbose=False, model="GP", **kwargs
    ):

        kwargs.pop("lhs_append_bounds", None)

        super(DLASED, self).__init__(
            bounds=bounds,
            n_init=n_init,
            rnd_state=rnd_state,
            verbose=verbose,
            optimizer=None,
            model=model,
            lhs_append_bounds=True,
            **kwargs
        )

        # additional param
        self._weight: float = 0.5
        self._X_cand = None
        self._prev_lcmodel = None  # init prev_lc model

        # create edge samples
        self._X_edge: np.ndarray = edge_sampling(self.bounds, self._design_space.n_dims)

    def _find_support_points(self, X_cand: np.ndarray, X_obs: np.ndarray):
        """
        Find the support points for candidate points based on triangulation

        Returns
        -------
        candidate_sp: np.ndarray
            Support points for each candidate. Stored in an array with
            shape (k, dims+1, dims)

        candidate_spcenter: np.ndarray
            Circumcenter of the sphere for each candidate.  Stored in an array with
            shape (k, dims)

        candidate_spradius: np.ndarray
            Circumradius of the sphere for each candidate.  Stored in an array with
            shape (k, dims)
        """
        dims = self._design_space.n_dims
        tri = None

        if dims == 1:  # don't use triangulation if dim=1

            # get permutations
            support_points_indices = list(
                permutations(np.arange(0, X_obs.shape[0]), r=2)
            )
            support_points_indices = np.unique(
                np.sort(support_points_indices, axis=1), axis=0
            )

            # get combinations of points (n_sp out of u)
            support_points = X_obs[support_points_indices, :]

            # calc sphere center and radius
            sphere_centers = np.mean(support_points[:, :, 0], axis=1)
            sphere_radius = np.abs(support_points[:, 0, 0] - sphere_centers)
            sphere_centers = sphere_centers.reshape(-1, 1)

            # Step 2: Exclude spheres which including other design points

            # drop spheres where other points are included
            index_j = 0
            hold_spheres = []
            for center_j, radius_j, support_p in zip(
                sphere_centers, sphere_radius, support_points_indices
            ):  # iter over spheres

                # drop support points (could be included because of rounding errors)
                x_obs = np.delete(X_obs, support_p, axis=0)

                # calc distance between x and sphere center
                distance = np.linalg.norm(x_obs - center_j, axis=1)

                # append index if sphere has no points within radius
                if (distance >= radius_j).all():
                    hold_spheres.append(index_j)
                index_j += 1

            # delete spheres where other points of X are included
            support_points = support_points[hold_spheres, :, :]
            support_points_indices = support_points_indices[hold_spheres, :]
            sphere_centers = sphere_centers[hold_spheres, :]
            sphere_radius = sphere_radius[hold_spheres]

            # parallelized sphere selection
            candidate_spindex = Parallel(n_jobs=self.n_jobs)(
                delayed(self._select_sphere)(
                    dims, x_cand_i, support_points, sphere_centers, sphere_radius, tri
                )
                for x_cand_i in X_cand
            )

        else:  # if dim > 1 -> use delaunay triangulation
            # Step 1: Calculate all the center of sphere O_j defined with arbitrary
            # dims + 1 points of X = [x_1, ..., x_u]
            tri = Delaunay(X_obs)
            support_points_indices = tri.simplices  # shape=(n_simplex, n_dim+1)
            support_points = X_obs[
                support_points_indices
            ]  # shape=(n_simplex, n_dim+1, n_dim)

            # Step 3: Select the spheres including alternative point x_a according to the
            # distance between x_a and O_j
            candidate_spindex = tri.find_simplex(X_cand)  # get simplex containing point

            sphere_radius = np.zeros(support_points.shape[0])  # only needed if dim == 1

        # resulting support points for each candidate
        candidate_spindex: np.ndarray = np.array(candidate_spindex)

        candidate_sp = support_points[candidate_spindex, :, :]
        cand_support_points_indices = support_points_indices[candidate_spindex, :]

        if candidate_spindex.min() == -1:
            print("Warning: Some candidate points are not contained in simplex!")

        return [
            candidate_spindex,
            candidate_sp,
            support_points,
            sphere_radius,
            cand_support_points_indices,
        ]

    def _select_sphere(
        self, dims: int, x_cand_i, support_points, sphere_centers, sphere_radius, tri
    ):
        """
        Select sphere for given candidate point

        Returns
        -------
        sphere_index: int
            selected sphere for given candidate point
        """
        # Step 4: candidate point is included in the constructed polyhedron
        if dims == 1:

            tt_min = np.min(support_points[:, :, 0], axis=1) <= x_cand_i
            tt_max = x_cand_i <= np.max(support_points[:, :, 0], axis=1)
            sp_ids = np.logical_and(tt_min, tt_max)

            if np.all(sp_ids):
                raise ValueError("Point not included: %s" % x_cand_i)
        else:
            sp_ids = tri.find_simplex(
                [x_cand_i]
            )  # returns only one sp_ids even if cand. point is included in more

        # Step 5: select sphere with max radius
        rad = sphere_radius.copy()

        mask = np.ones(len(rad), np.bool)
        mask[sp_ids] = 0
        rad[mask] = -1
        selected_sphere = np.argmax(rad)

        return selected_sphere

    def _calc_vol(self, sphere_supports, sphere_rads) -> np.ndarray:
        """
        Calculate volume of polyhedron with sphere support points and sphere radius
        """
        # dim of design space
        m: int = self._design_space.n_dims

        poly_vol = list()

        # calculate volume of support
        prnt_warning: bool = False

        for sphere_sp, sphere_rad in zip(sphere_supports, sphere_rads):

            if m == 1:  # calc area of circle
                vol = 4 * np.pi * sphere_rad**2

            elif (
                m <= 10
            ):  # calc area(2-dim) or volume (>2-dim) of polyhedron created by support
                # points
                try:
                    hull = ConvexHull(sphere_sp)
                    vol = hull.volume
                except Exception:
                    prnt_warning = True
                    vol = 0

            else:  # use volume approximation
                D = np.abs(sphere_sp[None, :] - sphere_sp[:, None])
                vol = np.prod(np.max(D, axis=(0, 1)))

            poly_vol.append(vol)

        if prnt_warning:
            print(
                "Warning: QH6154 Qhull precision error: Initial simplex is flat "
                "(facet 1 is coplanar with the interior point)"
            )

        return np.array(poly_vol)

    def _calc_explor(
        self, candidate_support: list, X_cand: np.ndarray, y_obs: np.ndarray
    ) -> np.ndarray:
        """
        Calculate global exploration value g(x_a, X)

        :return:

        cand_exploration: np.ndarray
            global exploration value for candidate points
        """
        # dim of design space
        m = self._design_space.n_dims

        # get prediction from surrogate model
        y_pred = self.surr_model.predict(X_cand)
        y_pred = np.tile(y_pred.flatten(), (m + 1, 1)).T

        # get observed y at support points
        y_obs_sp = y_obs[candidate_support[4]]

        # calculate euclidean distance between prediction and support point prediction
        dist1 = np.abs(y_pred - y_obs_sp)

        # calculate euclidean distance between candidate point x_a and it's support
        # points x_ki
        x_cand = np.tile(X_cand, (m + 1, 1, 1))
        x_cand = np.transpose(x_cand, (1, 0, 2))

        dist2 = np.linalg.norm(x_cand - candidate_support[1], axis=2)

        # calculate volume only for used spheres
        used_spheres_id, un_index = np.unique(
            candidate_support[0], return_index=True
        )  # get id from used spheres
        # calc volume only from used spheres
        poly_vol = self._calc_vol(
            candidate_support[2][used_spheres_id], candidate_support[3][used_spheres_id]
        )
        # reconstruct with inverse ids and get volume for candidates
        cand_vol = np.zeros(candidate_support[2].shape[0])
        cand_vol[used_spheres_id] = poly_vol
        cand_vol = cand_vol[candidate_support[0]]

        # calculate global exploration value
        cand_exploration = cand_vol * np.product(dist2 * dist1, axis=1)

        if self.verbose:
            print("ArgMax:", np.argmax(cand_exploration))

        return cand_exploration

    def _calc_exploi(self, X_cand: np.ndarray, X_obs: np.ndarray, y_obs: np.ndarray):
        """
        Calculate Leave one out Cross validation (LC) error and train model

        Parameters
        ----------
        X_cand: np.ndarray, shape=(n_candidates, n_features)
        X_obs: np.ndarray, shape=(n_observations, n_features)
        y_obs: np.ndarray, shape=(n_observations)

        Returns
        -------
        candidate_l_expl: np.ndarray, shape=(n_candidates)
            predicted error for candidate points
        model_perr: SurrModel
            fitted model to prediction error for unobserved points
        """
        # init error prediction model
        model_perr: SurrogateRegressionModel = cook_model(
            "GP", rnd_state=self._rnd_state
        )

        # calc LOO error
        lc_error_v = self.surr_model.loo()

        # fit model to prediction error for unobserved points
        model_perr.fit(X_obs, lc_error_v.flatten())

        # predict error for candidate points
        candidate_l_expl = model_perr.predict(X_cand).flatten()

        return candidate_l_expl, model_perr

    def _acquisition_fun(
        self, X_cand: np.ndarray, X_obs: np.ndarray, y_obs: np.ndarray
    ):

        candidate_support = self._find_support_points(X_cand, X_obs)

        # calculate global exploration value
        cand_g_expl = self._calc_explor(candidate_support, X_cand, y_obs)

        # calculate local exploration value
        cand_l_expl, lc_model = self._calc_exploi(X_cand, X_obs, y_obs)

        # calculate acquisition value for candidates
        aq_lc = self._weight * cand_l_expl / np.max(cand_l_expl)
        aq_gl = (1 - self._weight) * cand_g_expl / np.max(cand_g_expl)
        aq_cand = aq_lc + aq_gl

        if self.verbose:
            print(
                "Exploration: %.2f, Exploitation: %.2f"
                % ((1 - self._weight), self._weight)
            )

        return aq_cand, lc_model

    def _update_weight(self, lc_model, lc_model_prev, X_obs: np.ndarray) -> float:
        """
        Adaptively update weights
        """
        m: int = self._design_space.n_dims

        # init value
        weight: float = 0.5

        # update weight
        if lc_model_prev is not None:
            x_i = X_obs[-1].reshape(-1, m)

            # local improvement
            p = np.power(lc_model.predict(x_i)[0], 2) / np.power(
                lc_model_prev.predict(x_i)[0], 2
            )

            # global improvement
            q = np.sum(lc_model.predict(X_obs) ** 2) / np.sum(
                lc_model_prev.predict(X_obs) ** 2
            )

            # update weight
            weight = np.min([0.5 * p / q, 1])

        return weight

    def _return_aq(self, x):
        return [self._X_cand, self._aq_cand]

    def _check_edge_samples(self, X_obs: np.ndarray) -> Optional[List[float]]:
        """
        Check if edges are already sampled, if not propose next samples

        Returns
        -------
        next_sample: np.ndarray or None, shape=(n_samples, n_features)
            next point to sample on edges, if all edges are sampled -> None is returned
        """
        next_sample: Optional[np.ndarray] = None

        for x_edge in self._X_edge:

            min_delta_x = min([self._design_space.distance(x_edge, xi) for xi in X_obs])

            if not abs(min_delta_x) <= 1e-4:  # x_edge is unknown -> next sample
                return x_edge

        return next_sample

    def ask_(self) -> np.ndarray:

        # get observations
        X_obs = np.array(self.X_observed)
        y_obs = np.array(self.y_observed).flatten()

        # sample on edges until all points are known, then proceed with adaptive approach
        next_x = self._check_edge_samples(
            X_obs
        )  # next edge point, if next_x=None -> all edges have been sampled

        if next_x is None:  # use adaptive sampling to propose next point
            # generate candidate points
            self._X_cand = self._gen_cand_points(
                "lhs", n_samples=5000 * self._design_space.n_dims, lhs_crit=None
            )

            # calculate acquisition for candidate points
            self._aq_cand, lc_model = self._acquisition_fun(self._X_cand, X_obs, y_obs)

            # find new candidate with max aq value
            next_x_index = np.argmax(self._aq_cand)
            next_x = self._X_cand[next_x_index]

            # track information
            self.tracking_i["alpha"] = self._weight

            # update weight
            self._weight = self._update_weight(lc_model, self._prev_lcmodel, X_obs)

            # assign previous lc model
            self._prev_lcmodel = deepcopy(lc_model)

        return next_x
