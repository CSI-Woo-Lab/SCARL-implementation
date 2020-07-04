import numpy as np
import math

class JobGenerator(object):
    def __init__(self, lamb_min=2.0, lamb_max=4.0, res_dists=None, len_dists=None, resample_cycle=[20, 5, 10], simple=False):
        self.lamb_min = lamb_min
        self.lamb_max = lamb_max

        self.gpu_m = [8, 8, 1]
        self.gpu_M = [16, 16, 2]
        self.cpu_i_m = [8, 4, 0]
        self.cpu_i_M = [12, 6, 0]
        self.mem_i_m = [4, 8, 0]
        self.mem_i_M = [6, 12, 0]
        self.soso_m = [4, 4, 0]
        self.soso_M = [12, 12, 0]
        self.resample_cycle = resample_cycle
        self.simple = simple
        self.jobs = [
            [self.gpu_m, self.gpu_M],
            [self.cpu_i_m, self.cpu_i_M],
            [self.mem_i_m, self.mem_i_M],
            [self.soso_m, self.soso_M],
        ]
        self.sh_vecs = [
            [0, 0], # 1
            [0, 0], # 2
            [0, 0], # 3
            [0, 0], # 4
            [0, 0], # 5
            [1, 1], # 6
            [1, 0], # 7
            [1, 0], # 8
            [0, 1], # 9
            [0, 1], # 10
        ]

        self.gamma_short = 0.15
        self.gamma_long = 0.05
        self.short_m, self.short_M = 1, 30
        self.long_m, self.long_M = 40, 100

        self.res_dists = [
            np.array([0.6, 0.4]),
            np.array([0.5, 0.5]),
            np.array([0.4, 0.6]),
            np.array([0.2, 0.8])
        ]

        self.len_dists = [
            np.array([0.6, 0.4]),
            np.array([0.5, 0.5]),
            np.array([0.4, 0.6]),
            np.array([0.2, 0.8])
        ]

        self.job_portion = self.job_portion_calc()
        self.res_dist = self.res_dists[0]
        self.len_dist = self.len_dists[0]
        self.job_dist_recalc_counter = 0
        self.len_dist_recalc_counter = 0
        self.res_dist_recalc_counter = 0
        self.reset()

    def reset(self):
        self.job_dist = self.job_portion_calc()
        self.res_dist = self.res_dists[0]
        self.len_dist = self.len_dists[0]
        self.job_dist_recalc_counter = 0
        self.len_dist_recalc_counter = 0
        self.res_dist_recalc_counter = 0

    def job_portion_calc(self, n=4):
        cnt = 0
        if self.simple:
            return (np.ones(n) / n)
        while True:
            prob_vec = np.random.randint(1, 11, n)
            for x in range(len(prob_vec)):
                if prob_vec[x] <= 7:
                    prob_vec[x] = 0
                elif prob_vec[x] == 10:
                    prob_vec[x] = 2
                else:
                    prob_vec[x] = 1
            #print(prob_vec)
            if prob_vec.sum() > 0:
                break

        prob_vec = prob_vec / prob_vec.sum()
        return prob_vec

    def get_new_job(self,):
        job_i = np.random.choice(len(self.job_portion), p=self.job_portion)
        job_min, job_max = self.jobs[job_i]
        r = np.random.rand()
        if r >= self.res_dist[0]:
            job_min = [x * 2 for x in job_min]
            job_max = [x * 2 for x in job_max]
        job_j = np.random.choice(len(self.sh_vecs))
        ex_vec = np.array([np.random.randint(minv, 1 + maxv) for (minv, maxv) in zip(job_min, job_max)])
        sh_vec = self.sh_vecs[job_j]

        r = np.random.rand()
        if r >= self.len_dist[0]:
            dur_gamma = self.gamma_long
            dur_min, dur_max = self.long_m, self.long_M
        else:
            dur_gamma = self.gamma_short
            dur_min, dur_max = self.short_m, self.short_M
        dur = (dur_min - 1) + np.random.geometric(dur_gamma)
        if dur < dur_min:
            dur = dur_min
        elif dur > dur_max:
            dur = dur_max
        disc = np.random.normal(12.0, 6.0)

        job_boundary = math.ceil(disc + dur)
        job_boundary = min(50, job_boundary)
        job_boundary = max(dur, job_boundary)

        return (dur, -1, ex_vec, sh_vec, 0, job_boundary)

    def mmpp(self):

        if self.job_dist_recalc_counter <= 0:
            self.job_portion = self.job_portion_calc()
            self.job_dist_recalc_counter = 10 + np.random.geometric(0.05)
            #print("job_dist transition!", self.job_dist_recalc_counter)
            if self.len_dist_recalc_counter < self.resample_cycle[0]:
                self.len_dist_recalc_counter = self.resample_cycle[0]

        if self.len_dist_recalc_counter <= 0:
            self.len_dist = self.len_dists[np.random.randint(0, len(self.len_dists))]
            self.len_dist_recalc_counter = np.random.geometric(0.08)
            if self.len_dist_recalc_counter < self.resample_cycle[1]:
                self.len_dist_recalc_counter = self.resample_cycle[1]

        if self.res_dist_recalc_counter <= 0:
            self.res_dist = self.res_dists[np.random.randint(0, len(self.res_dists))]
            self.res_dist_recalc_counter = 5 + np.random.geometric(0.08)
            if self.res_dist_recalc_counter < self.resample_cycle[2]:
                self.res_dist_recalc_counter = self.resample_cycle[2]
        lamb = self.lamb_min + (self.lamb_max - self.lamb_min) * np.random.random()
        job_count = np.random.poisson(lamb)
        ret = [self.get_new_job() for _ in range(job_count)]

        self.job_dist_recalc_counter = self.job_dist_recalc_counter - 1
        self.len_dist_recalc_counter = self.len_dist_recalc_counter - 1
        self.res_dist_recalc_counter = self.res_dist_recalc_counter - 1
        return ret
