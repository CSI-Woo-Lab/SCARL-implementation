from queue import Queue
import numpy as np
import math
import gym
from numba import jit


class Env(gym.Env):
    def __init__(self, max_simu_len, machine_descriptions,
                 seq=None, n_machines=3, render=False,
                 reward_type='completion_time',
                 with_deadline=False,
                 with_dur=False,
                 with_boundary=False,
                 time_horizon=200):

        super(Env, self).__init__()

        self.max_simu_len = max_simu_len
        self.n_machines = n_machines
        self.rew_avg = 0.0
        self.reward_type = reward_type
        self.seq = seq
        self.with_dur = with_dur
        self._render = render
        self.curr_time = 0
        self.time_horizon = time_horizon
        self.seq_no = 0  # which example sequence
        self.seq_idx = 0  # index in that sequence
        self.curr_simu_len = 0
        self.with_boundary = with_boundary
        # initialize system
        self._job_id = 0
        self.machine_descriptions = machine_descriptions
        self.with_deadline = with_deadline
        self.init_machine()
        self.job_slot = JobSlot()
        self.allocable_jobs = []
        self.impossible_jobs = []
        self.job_record = dict()
        self.utils_slot = []
        self.job_queue = Queue()

    def init_machine(self):
        self.machines = []
        for desc in self.machine_descriptions:
            machine = Machine(self.time_horizon, desc, self.with_dur, self.with_boundary)
            self.machines.append(machine)

    #@jit
    def slot_decode(self):
        jobs = np.array([job.as_vec() for job in self.job_slot.slot])
        machines = np.array([machine.as_vec() for machine in self.machines])
        return jobs, machines

    @jit
    def observe(self):
        jobs, machines = self.slot_decode()
        allocable_jobs = self.get_allocable_jobs()
        allocable_machines = self.get_allocable_machines_per_job()
        return jobs, machines, allocable_jobs, allocable_machines

    def get_reward(self, elapsed_time, deleted_jobs):
        if self.with_deadline is True:
            return -deleted_jobs

        if self.reward_type == 'completion_time':
            jobs_in = 0
            for machine in self.machines:
                #jobs_in += len(machine.running_job)
                for job in machine.running_job:
                    jobs_in += (1 / job.len)
            jobs_in += len(self.job_slot.slot)
            return -jobs_in * elapsed_time
        elif self.reward_type == 'utilization':
            raise NotImplementedError("Not implemented yet, TODO: implement this!")
            """
            threshold = self.pa.res_slot * (1.0 - 0.7)
            rew = -0.1 * np.sum((np.array(self.utils_slot) - threshold) ** 2) / self.pa.num_res
            self.utils_slot = []
            return rew
            """

        raise EnvironmentError("arrived that shouldn't be realised")

    def is_done(self):
        return (self.curr_time >= self.curr_simu_len) and len(self.job_slot.slot) == 0

    #@jit()
    def step(self, a):
        prev_step = self.curr_time
        done = False
        job_id, machine_id = a
        if job_id == -1 or machine_id == -1:
            self.time_proceed()
            allocated = False
        else:
            job = self.job_slot.slot[job_id]
            allocated = self.machines[machine_id].allocate_job(
                job,
                self.curr_time,
                self.with_deadline)
            if allocated:
                self.job_slot.slot.pop(job_id)


        done, deleted_jobs = self.get_time_through()

        elapsed_time = (self.curr_time - prev_step)

        return self.observe(), self.get_reward(elapsed_time, deleted_jobs), done, (allocated, elapsed_time)

    #@jit()
    def get_time_through(self):
        done = self.is_done()
        deleted_jobs = 0
        self.allocable_jobs = []
        while (len(self.allocable_jobs) == 0) and (not done):
            #print("!?!")
            self.allocable_jobs = []
            """
            if self.with_deadline:
                jobs = []
                for i, j in enumerate(self.job_slot.slot):
                    if j.len + self.curr_time < j.deadline:
                        jobs.append(j)
                    else:
                        deleted_jobs += 1
                self.job_slot.slot = jobs
            """
            for i, j in enumerate(self.job_slot.slot):
                for m in self.machines:
                    if m.is_allocable(j, self.curr_time, self.with_deadline):
                        self.allocable_jobs.append(i)
                        break

            if len(self.allocable_jobs) == 0:
                self.time_proceed()

            done = self.is_done()
            if done:
                break
        #print("!?!")
        if done:
            while True:
                running_jobs = 0
                for machine in self.machines:
                    running_jobs += len(machine.running_job)
                #print("running jobs", running_jobs)
                if running_jobs == 0:
                    break
                else:
                    self.time_proceed()
                #print(self.curr_time)
            return done, deleted_jobs
        return False, deleted_jobs

    @jit()
    def get_allocable_machines_per_job(self, ):
        ret = []
        for i, job in enumerate(self.job_slot.slot):
            temp = []
            for j, m in enumerate(self.machines):
                if m.is_allocable(job, self.curr_time, self.with_deadline):
                    temp.append(j)
            ret.append(temp)

        return ret



    def get_allocable_jobs(self):
        return self.allocable_jobs


    def render(self, mode='human', close=False):
        pass

    @jit()
    def get_new_job(self, raw_job):
        new_job = Job(raw_job, job_id=self.new_job_id(), enter_time=self.curr_time, with_dur=self.with_dur, with_boundary=self.with_boundary)
        self.job_queue.put(new_job)
        self.job_slot.slot.append(new_job)
        self.job_record[new_job.id] = new_job
        return new_job

    @jit()
    def time_proceed(self):
        self.curr_time += 1
        for m in self.machines:
            m.time_proceed(self.curr_time)

        if (self.curr_time - 1 < self.curr_simu_len):
            if self.curr_time - 1 < len(self.seq[self.seq_no]):
                for raw_job in self.seq[self.seq_no][self.curr_time - 1]:
                    new_job = self.get_new_job(raw_job)

    def new_job_id(self):
        ret = self._job_id
        self._job_id += 1
        return ret

    @jit()
    def reset(self, seq_no=0, curr_simu_len=100):
        self.seq_idx = 0
        self.curr_time = 0
        self._job_id = 0
        self.job_queue = Queue()
        self.job_record = dict()
        self.job_slot = JobSlot()
        self.curr_simu_len = min(curr_simu_len, self.max_simu_len)
        self.seq_no = seq_no


        # initialize system
        self.init_machine()

        self.allocable_jobs = np.arange(len(self.job_slot.slot)).tolist()

        self.seq_idx = 0
        self.get_time_through()
        return self.observe()


class JobSlot:
    def __init__(self):
        self.slot = []


class Job(object):
    def __init__(self, raw_job, job_id, enter_time, with_dur=False, with_boundary=False):
        self.id = job_id,
        self.len = raw_job[0]
        self.job_dur_type = raw_job[1]
        self.ex_res = raw_job[2]
        self.sh_res = raw_job[3]
        self.enter_time = enter_time
        self.deadline = enter_time + (self.len * 3)
        self.start_time = -1 # not allocated yet
        self.finish_time = -1
        self.ex_max = np.array([48, 48, 4])
        self.with_dur = with_dur
        self.with_boundary = with_boundary
        if self.with_boundary:
            self.boundary = raw_job[5]

    def as_vec(self):
        ex = self.ex_res / self.ex_max[:len(self.ex_res)]
        if self.with_dur is False:
            return np.hstack([ex, 1 - ex, self.sh_res])
        else:
            lseq = (self.len / 50)
            if self.with_boundary:
                lseq = (self.boundary / 50)

            return np.hstack([lseq, 1 - lseq, ex, 1 - ex, self.sh_res])

    """
    def __repr__(self):
        return "job_id: %d\t enter_time: %d\tstart_time: %d\tfinish_time: %d\tlen %d\n" % (self.id, self.enter_time, self.start_time, self.finish_time, self.len)

    def __str__(self):
        return self.__repr__()
    """


class Machine(object):
    def __init__(self, time_horizon, desc, with_dur, with_boundary):
        """
        exclusive resource: CPU, GPU, and any other hard constraint;
        어느 잡이 사용중일 땐 다른 잡은 쓸 수 없는 리소스
        sharable resource: SSD/ CPU 속도:
        어느 잡이 사용 중일때에도 다른 잡이 쓸 수 있는 리소스:
        특히, CPU의 속도가 빠르다던가, 하는 특징을 binary로 나타낼 수 있음.
        """
        self.time_horizon = time_horizon
        self.ex_capacity = desc['ex_res']
        self.sh_capacity = desc['sh_res']
        self.ex_max = desc['ex_max']
        self.type = desc['type']
        self.num_types = 2
        if 'num_types' in desc:
            self.num_types = desc['num_types']
        self.type_vec = np.zeros(self.num_types, np.float32)
        self.type_vec[self.type - 1] = 1.0
        self.with_dur = with_dur
        self.with_boundary = with_boundary

        self.ex_vec = (np.ones((self.time_horizon, len(self.ex_capacity)), dtype=np.int32) * np.array(self.ex_capacity)).astype(np.int32)

        self.ex_div = np.array(self.ex_max, dtype=np.float32) + np.float32(1e-9)
        self.sh_vec = np.array(self.sh_capacity, dtype=np.int32)
        self.running_job = []
        self.estimate_usage = np.zeros((2, len(self.ex_capacity)), dtype=np.int32)

    def as_vec(self):
        res = (self.ex_vec[0, :] / self.ex_div).ravel()

        return np.hstack([
            res,
            1.0 - res,
            self.sh_capacity,
            self.type_vec]).ravel()

    def is_allocable(self, job, curr_time, with_deadline):
        if with_deadline:
            act_len = job.len
            if self.type == 2:
                act_len = math.ceil(act_len * 1.5)
            if self.type == 3:
                act_len = math.ceil(act_len * 2)

            ret = (job.deadline >= (curr_time + job.len)) and \
                  (np.all(self.sh_vec - job.sh_res >= 0) and np.all((self.ex_vec[0, :] - job.ex_res) >= 0)) and \
                  (job.finish_time <= curr_time + act_len)
        else:
            ret = (np.all(self.sh_vec - job.sh_res >= 0) and np.all((self.ex_vec[0, :] - job.ex_res) >= 0))
        return ret

    def allocate_job(self, job, curr_time, with_deadline):
        allocated = False

        if self.is_allocable(job, curr_time, with_deadline):
            act_len = job.len

            if self.type == 2:
                act_len = math.ceil(act_len * 1.5)
            if self.type == 3:
                act_len = math.ceil(act_len * 2)

            self.ex_vec[:act_len, :] -= job.ex_res

            job.start_time = curr_time
            job.finish_time = curr_time + act_len
            self.running_job.append(job)
            allocated = True
        return allocated

    def time_proceed(self, curr_time):
        self.ex_vec[:-1, :] = self.ex_vec[1:, :]
        self.ex_vec[-1, :] = self.ex_capacity

        for job in self.running_job:
            elapsed = curr_time - job.start_time

            if job.finish_time <= curr_time:
                self.running_job.remove(job)
