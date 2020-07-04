import numpy as np
import math

def get_TETRIS_M(env, j, allocable_machines, with_deadline=False):
    curr_min_dur, curr_j, curr_m = 100000, -1, -1

    for m in allocable_machines[j]:
        dur = -np.dot(env.machines[0].ex_vec[0, :], env.job_slot.slot[j].ex_res)
        if env.machines[m].is_allocable(env.job_slot.slot[j], env.curr_time, with_deadline) and  dur < curr_min_dur:
            curr_min_dur = dur
            curr_m = m

    return curr_m

def get_RR_M(env, j, allocable_machines, last_m, with_deadline=False):
    tt = allocable_machines[j] * 2
    idx = 0
    if last_m < max(allocable_machines[j]):
        while tt[idx] <= last_m:
            idx += 1
    else:
        idx = 0
    for _m in range(idx, len(tt)):
        m = allocable_machines[j][_m]
        #print(m)
        if env.machines[m].is_allocable(env.job_slot.slot[j], env.curr_time, with_deadline):
            return m
    return m

def get_RR_J(env, allocable_jobs, last_j):
    tt = allocable_jobs * 2
    idx = 0
    if last_j < max(allocable_jobs):
        while tt[idx] <= last_j:
            idx += 1
    else:
        idx = 0

    for _m in range(idx, len(tt)):
        j = allocable_jobs[_m]
        return j
    return j

def get_FIFO_J(env, allocable_jobs):
    return allocable_jobs[0]

def get_SJF_J(env, allocable_jobs):
    curr_min_dur, curr_j, curr_m = 100000, -1, -1
    for j in allocable_jobs:
        dur = env.job_slot.slot[j].len
        if env.with_boundary:
            dur = env.job_slot.slot[j].boundary
        if dur < curr_min_dur:
            curr_min_dur = dur
            curr_j = j
    return curr_j

def mc_FIFO_TETRIS(env, allocable_jobs, allocable_machines, last_j, last_m, with_deadline=False):
    j = get_FIFO_J(env, allocable_jobs)
    m = get_TETRIS_M(env, j, allocable_machines, env.curr_time, with_deadline)
    return j, m

def mc_FIFO_RR(env, allocable_jobs, allocable_machines, last_j, last_m, with_deadline=False):
    j = get_FIFO_J(env, allocable_jobs)
    m = get_RR_M(env, j, allocable_machines, last_m, env.curr_time, with_deadline)
    return j, m


def mc_SJF_RR(env, allocable_jobs, allocable_machines, last_j, last_m, with_deadline=False):
    curr_min_dur, curr_j, curr_m = 100000, -1, -1
    j = get_SJF_J(env, allocable_jobs)
    m = get_RR_M(env, j, allocable_machines, last_m, with_deadline)
    return (j, m)

def mc_SJF_TETRIS(env, allocable_jobs, allocable_machines, last_j, last_m):
    j = get_SJF_J(env, allocable_jobs)
    m = get_TETRIS_M(env, j, allocable_machines)
    return j, m

def mc_RR_RR(env, allocable_jobs, allocable_machines, last_j, last_m):
    j = get_RR_J(env, allocable_jobs, last_j)
    m = get_RR_M(env, j, allocable_machines, last_m)
    return j, m

def mc_RR_TETRIS(env, allocable_jobs, allocable_machines, last_j, last_m, with_deadline=False):
    j = get_RR_J(env, allocable_jobs, last_j)
    m = get_TETRIS_M(env, j, allocable_machines, with_deadline)
    return j,m
