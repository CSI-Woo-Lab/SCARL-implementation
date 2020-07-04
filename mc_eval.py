import pickle
import numpy as np
import torch
import torch.optim as optim
import os
import time
import scipy.signal
import mc_env
import torch.multiprocessing as mp
import networks
import argparse
import json

log_interval = 1


parser = argparse.ArgumentParser()
parser.add_argument("--reward_type", help="reward type, [slowdown, completion_time, utilization]", type=str, default="completion_time")
parser.add_argument("--save_dir", help="save directory", type=str)

parser.add_argument("--use_rnn", type=bool, default=False)
parser.add_argument("--min_sim_len", type=int, default=30)
parser.add_argument("--max_sim_len", type=int, default=50)
parser.add_argument("--step_size", type=int, default=20)
parser.add_argument("--load", help='load', type=int, default=-1)
parser.add_argument("--epochs", help='num_epocs', type=int, default=500)
parser.add_argument("--wo_attention", type=bool, default=False)
parser.add_argument("--boundary", type=bool, default=False)
parser.add_argument("--input", type=str, default="")
args = parser.parse_args()


save_dir = args.save_dir

torch.manual_seed(1541)
np.random.seed(1541)

with open('machines.json', 'r') as f:
    machine_desc = json.load(f)

def discount(x, gamma=1.00):
    return scipy.signal.lfilter([1],[1,-gamma],x[::-1], axis=0)[::-1]

def get_model():
    if args.load < 0:
        print("no load")
        print("use_rnn", args.use_rnn)
        return networks.attention_net(job_dim=10, machine_dim=10, embedding_size=32, wo_attention=args.wo_attention)
    else:
        print("load")
        with open(os.path.join(save_dir, "model_%d.out" % args.load), 'rb') as f:
            return torch.load(f)


if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def get_env(max_simu_len, seq):
    sz1 = 32
    sz2 = 32
    sz3 = 128
    sz4 = 16
    m_desc = [
        # machie with additional exclusive resources
        {'ex_res': [sz1, sz1, sz4], 'sh_res': [1, 1], 'type': 1, 'ex_max': [sz3, sz3, sz4]},
        {'ex_res': [sz2, sz3, 0], 'sh_res': [1, 0], 'type': 1, 'ex_max': [sz3, sz3, sz4]},
        {'ex_res': [sz3, sz2, 0], 'sh_res': [0, 1], 'type': 1, 'ex_max': [sz3, sz3, sz4]},
        {'ex_res': [sz1, sz1, 0], 'sh_res': [1, 1], 'type': 2, 'ex_max': [sz3, sz3, sz4]},
        {'ex_res': [sz1, sz1, 0], 'sh_res': [1, 1], 'type': 1, 'ex_max': [sz3, sz3, sz4]},
    ]

    return mc_env.Env(max_simu_len, seq=seq, machine_descriptions=m_desc, with_dur=True, with_boundary=args.boundary)


def rollout(seq_no, curr_simu_len, worker, env, param_queue, batch_size=20):
    worker.reset()
    rews = []
    logpasses = []
    returns = []
    baselines = np.zeros((batch_size, 5000))
    worker.zero_grad()
    for batch in range(batch_size):

        r = 0
        #print(seq_no, curr_simu_len)
        ob = env.reset(seq_no=seq_no, curr_simu_len=curr_simu_len)
        worker.reset()
        logpas = []
        t_rews = []
        t_logp = 0
        for ep in range(5000):
            jobs, machines, allocable_job_list, allocable_machine_list = ob
            jobs = torch.from_numpy(jobs).float()
            #print(jobs.shape)
            machines = torch.from_numpy(machines).float()
            (j, m), logp = worker(jobs, machines, allocable_job_list, allocable_machine_list, argmax=False)
            ob2, rew, done, (alloc, elapsed_time) = env.step((j, m))

            t_logp += logp
            if elapsed_time > 0:
                logpas.append(t_logp)
                t_rews.append(rew / 10.0)
                t_logp = 0
            r += rew
            if done:
                rew = r / 10.0
                logpasses.append(torch.stack(logpas))
                rews.append(r)
                break
            ob = ob2

        discounted_returns = discount(t_rews, 1.00)
        returns.append(discounted_returns)
        baselines[batch, :len(discounted_returns)] = discounted_returns

    baseline = np.mean(baselines, axis=0)
    rews = np.array(rews)
    l = 0.0
    for i in range(batch_size):
        R = returns[i]
        b = logpasses[i]
        a = torch.from_numpy((R - baseline[:len(R)])).float()
        l += torch.sum(-(a * b))

    l /= batch_size
    l.backward()
    grads = [x.grad for x in worker.parameters()]
    ret = (grads, np.array(rews), l.detach().numpy())
    param_queue.put(ret)
    return ret


def main():
    num_worker = 4
    num_simulation = 100
    min_simu_len = args.min_sim_len
    max_simu_len = args.max_sim_len

    with open(os.path.join("data", args.input), 'rb') as f:
        seqs = pickle.load(f)

    list_of_grads = []
    running_reward = None
    param_queue = mp.Queue()
    torch.multiprocessing.set_sharing_strategy('file_system')
    envs = [get_env(max_simu_len, seqs) for x in range(num_worker)]

    workers = [get_model() for _ in range(num_worker)]

    global_policy = get_model()
    global_policy.train()

    for w in workers:
        w.load_state_dict(global_policy.state_dict())
    for w in workers:
        w.train()
    global_opt = optim.RMSprop(global_policy.parameters(), lr=1e-3)
    print("training begins!")
    prev_time = time.time()
    for i_episode in range(args.load + 1, args.epochs):
        ps = []
        simlens = np.random.randint(min_simu_len, max_simu_len + 1, num_worker)
        seq_indices = np.random.choice(num_simulation, num_worker, replace=True)
        for i in range(num_worker):
            simlens[i] = min(simlens[i], max_simu_len)
            #print(simlens[i], max_simu_len)
            p = mp.Process(
                target=rollout,
                args=(seq_indices[i], simlens[i], workers[i], envs[i], param_queue, args.step_size))
            ps.append(p)
        for p in ps:
            p.start()

        #for p in ps:
        #    p.join()

        list_of_grads = []
        rewards = []
        losses = []
        global_opt.zero_grad()
        tot_loss = 0
        for i in range(num_worker):
            g, r, l = param_queue.get()
            list_of_grads.append(g)
            rewards.append(r)
            losses.append(l)
            tot_loss += l

        ep_reward = np.mean(rewards)
        if running_reward is None:
            running_reward = ep_reward
        else:
            running_reward = 0.1 * ep_reward + (1 - 0.1) * running_reward
        torch.nn.utils.clip_grad_norm_(global_policy.parameters(), 1.5)
        ep_reward = np.mean(rewards)
        global_opt.zero_grad()
        for grads in list_of_grads:
            for p, g in zip(global_policy.parameters(), grads):
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad += g

        for p in global_policy.parameters():
            if p.grad is None:
                continue
            p.grad /= len(list_of_grads)
        global_opt.step()
        for w in workers:
            w.load_state_dict(global_policy.state_dict())

        if i_episode % log_interval == 0:
            curr_time = time.time()
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward), 'losses %0.2f' % np.mean(losses),
                'elapsed time', curr_time - prev_time)
            prev_time = curr_time
            with open(os.path.join(save_dir, "model_%d.out" % i_episode), 'wb') as f:
                torch.save(global_policy, f)

    curr_time = time.time()
    print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
        i_episode, ep_reward, running_reward), 'losses %0.2f' % np.mean(losses),
        'elapsed time', curr_time - prev_time)
    prev_time = curr_time
    with open(os.path.join(save_dir, "model_%d.out" % i_episode), 'wb') as f:
        torch.save(global_policy, f)

if __name__ == '__main__':
    main()
