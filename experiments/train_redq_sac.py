import gymnasium as gym
import numpy as np
import torch
import time
import sys
import os
import re
from redq.algos.redq_sac import REDQSACAgent
from redq.algos.core import mbpo_epoches, test_agent
from redq.utils.run_utils import setup_logger_kwargs
from redq.utils.bias_utils import log_bias_evaluation
from redq.utils.logx import EpochLogger

def redq_sac(env_name, seed=0, epochs='mbpo', steps_per_epoch=1000,
             max_ep_len=1000, n_evals_per_epoch=1,
             logger_kwargs=dict(), debug=False, n_episodes_per_eval=10,
             # following are agent related hyperparameters
             hidden_sizes=(256, 256), replay_size=int(1e6), batch_size=256,
             lr=3e-4, gamma=0.99, polyak=0.995,
             alpha=0.2, auto_alpha=True, target_entropy='auto',
             start_steps=5000, delay_update_steps='auto',
             utd_ratio=20, num_Q=10, num_min=2, q_target_mode='min',
             policy_update_delay=20,
             # following are bias evaluation related
             evaluate_bias=True, n_mc_eval=1000, n_mc_cutoff=350, reseed_each_epoch=True,
             # Add resume path argument
             resume_path=None
             ):
    """
    :param env_name: name of the gym environment
    :param seed: random seed
    :param epochs: number of epochs to run
    :param steps_per_epoch: number of timestep (datapoints) for each epoch
    :param max_ep_len: max timestep until an episode terminates
    :param n_evals_per_epoch: number of evaluation runs for each epoch
    :param logger_kwargs: arguments for logger
    :param debug: whether to run in debug mode
    :param hidden_sizes: hidden layer sizes
    :param replay_size: replay buffer size
    :param batch_size: mini-batch size
    :param lr: learning rate for all networks
    :param gamma: discount factor
    :param polyak: hyperparameter for polyak averaged target networks
    :param alpha: SAC entropy hyperparameter
    :param auto_alpha: whether to use adaptive SAC
    :param target_entropy: used for adaptive SAC
    :param start_steps: the number of random data collected in the beginning of training
    :param delay_update_steps: after how many data collected should we start updates
    :param utd_ratio: the update-to-data ratio
    :param num_Q: number of Q networks in the Q ensemble
    :param num_min: number of sampled Q values to take minimal from
    :param q_target_mode: 'min' for minimal, 'ave' for average, 'rem' for random ensemble mixture
    :param policy_update_delay: how many updates until we update policy network
    :param resume_path: Path to the checkpoint directory to resume training from.
    """
    if debug: # use --debug for very quick debugging
        hidden_sizes = [2,2]
        batch_size = 2
        utd_ratio = 2
        num_Q = 3
        max_ep_len = 100
        start_steps = 100
        steps_per_epoch = 100

    # use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # set number of epoch
    if epochs == 'mbpo' or epochs < 0:
        epochs = mbpo_epoches[env_name]
    total_steps = steps_per_epoch * epochs # Calculate total steps based on desired epochs

    """set up logger"""
    # If resuming, logger kwargs should ideally point to the original experiment directory
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    """set up environment and seeding"""
    env_fn = lambda: gym.make(env_name, max_episode_steps=max_ep_len)
    env, test_env, bias_eval_env = env_fn(), env_fn(), env_fn()
    # seed torch and numpy
    torch.manual_seed(seed)
    np.random.seed(seed)

    # seed environment along with env action space so that everything is properly seeded for reproducibility
    def seed_all(epoch):
        seed_shift = epoch * 9999
        mod_value = 999999
        env_seed = (seed + seed_shift) % mod_value
        test_env_seed = (seed + 10000 + seed_shift) % mod_value
        bias_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
        torch.manual_seed(env_seed)
        np.random.seed(env_seed)
        env.action_space.seed(env_seed)
        test_env.action_space.seed(test_env_seed)
        bias_eval_env.action_space.seed(bias_eval_env_seed)
    seed_all(epoch=0)

    test_env.reset(seed=seed)
    bias_eval_env.reset(seed=seed)

    """prepare to init agent"""
    # get obs and action dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    # if environment has a smaller max episode length, then use the environment's max episode length
    max_ep_len = env._max_episode_steps if max_ep_len > env._max_episode_steps else max_ep_len
    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    # we need .item() to convert it from numpy float to python float
    act_limit = env.action_space.high[0].item()
    # keep track of run time
    start_time = time.time()
    # flush logger (optional)
    sys.stdout.flush()

    intermediate_path = os.path.join(logger.output_dir, 'intermediate_models', logger.exp_name)
    if not os.path.exists(intermediate_path):
        os.makedirs(intermediate_path)
    final_agent_path = os.path.join(logger.output_dir, 'models', logger.exp_name)
    if not os.path.exists(final_agent_path):
        os.makedirs(final_agent_path)
    #################################################################################################

    # Initialize start epoch and step
    start_epoch = 0
    start_step = 0

    """init agent"""
    agent = REDQSACAgent(env_name, obs_dim, act_dim, act_limit, device,
                 hidden_sizes, replay_size, batch_size,
                 lr, gamma, polyak,
                 alpha, auto_alpha, target_entropy,
                 start_steps, delay_update_steps,
                 utd_ratio, num_Q, num_min, q_target_mode,
                 policy_update_delay)

    """Load models if resuming"""
    if resume_path is not None and os.path.exists(resume_path):
        print(f"Resuming training from checkpoint: {resume_path}")
        start_epoch, start_step = agent.load_models(resume_path)
        # Reseed based on the loaded epoch to maintain consistency if reseed_each_epoch is True
        if reseed_each_epoch:
             print(f"Reseeding environment for epoch {start_epoch}")
             seed_all(start_epoch) # Seed based on the epoch we are starting *now*
    else:
        if resume_path is not None:
            raise ValueError(f"Resume path {resume_path} does not exist. Please check the path.")

    # Adjust start time calculation - note: this resets the timer upon resuming
    start_time = time.time()

    """start training"""
    (o, _), d, ep_ret, ep_len = env.reset(), False, 0, 0 # Reset env before starting loop

    # Adjust loop range if resuming
    print(f"Starting training loop from step {start_step} up to {total_steps}")
    for t in range(start_step, total_steps):
        # get action from agent
        a = agent.get_exploration_action(o, env) # Will take random action if needed

        # Step the env
        o2, r, term, trun, _ = env.step(a)
        d = term or trun
        ep_len += 1

        # Store data
        agent.store_data(o, a, r, o2, term) # Store term, not d
        # Update agent
        agent.train(logger)
        # Update obs
        o = o2
        ep_ret += r

        if d:
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            (o, _), d, ep_ret, ep_len = env.reset(), False, 0, 0

        # End of epoch wrap-up
        if (t + 1) % steps_per_epoch == 0:
            # Calculate epoch based on current step t
            # Epoch is completed at the *end* of the steps_per_epoch interval
            epoch = (t + 1) // steps_per_epoch - 1 # epoch 0 is completed after steps_per_epoch steps

            # Ensure we only run test/log/save if the epoch is >= the starting epoch
            if epoch >= start_epoch:
                # Test the performance
                # Printout testing time
                test_agent(agent, test_env, max_ep_len, logger, n_eval=n_episodes_per_eval)
                logger.store(TestEpTimestep=t+1) # Log the actual timestep
                logger.dump_eval()

                # Save intermediate checkpoint
                # Pass the completed epoch and step number
                agent.save_models(intermediate_path, epoch=epoch, total_steps_so_far=t)

                if evaluate_bias:
                    log_bias_evaluation(bias_eval_env, agent, logger, max_ep_len, alpha, gamma, n_mc_eval, n_mc_cutoff)

                # Reseed for the *next* epoch
                if reseed_each_epoch:
                    print(f"Reseeding environment for epoch {epoch + 1}")
                    seed_all(epoch + 1)

                """logging"""
                logger.log_tabular('Epoch', epoch)
                logger.log_tabular('TotalEnvInteracts', t)
                logger.log_tabular('Time', time.time()-start_time)
                logger.log_tabular('EpRet', with_min_and_max=True)
                logger.log_tabular('EpLen', average_only=True)
                logger.log_tabular('TestEpRet', with_min_and_max=True, clear=False)
                logger.log_tabular('TestEpLen', average_only=True, clear=False)
                logger.log_tabular('Q1Vals', with_min_and_max=True)
                logger.log_tabular('LossQ1', average_only=True)
                logger.log_tabular('LogPi', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('Alpha', with_min_and_max=True)
                logger.log_tabular('LossAlpha', average_only=True)
                logger.log_tabular('PreTanh', with_min_and_max=True)

                if evaluate_bias:
                    logger.log_tabular("MCDisRet", with_min_and_max=True)
                    logger.log_tabular("MCDisRetEnt", with_min_and_max=True)
                    logger.log_tabular("QPred", with_min_and_max=True)
                    logger.log_tabular("QBias", with_min_and_max=True)
                    logger.log_tabular("QBiasAbs", with_min_and_max=True)
                    logger.log_tabular("NormQBias", with_min_and_max=True)
                    logger.log_tabular("QBiasSqr", with_min_and_max=True)
                    logger.log_tabular("NormQBiasSqr", with_min_and_max=True)
                logger.dump_tabular()

                # flush logged information to disk
                sys.stdout.flush()

    # Save the final model parameters
    # Calculate final epoch and step
    final_epoch = (total_steps // steps_per_epoch) -1
    final_step = total_steps - 1
    agent.save_models(final_agent_path, epoch=final_epoch, total_steps_so_far=final_step)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Hopper-v5')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=-1) # -1 means use mbpo epochs
    parser.add_argument('--exp_name', type=str, default='redq')
    parser.add_argument('--data_dir', type=str, default='../data/')
    parser.add_argument('--debug', action='store_true')
    # Add resume path argument to parser
    parser.add_argument('--resume_dir', type=str, default=None, help="Path to checkpoint directory to resume from.")
    args = parser.parse_args()

    # Check if the resume   
    if args.resume_dir is not None:
        resume_model_directory = os.path.join(args.resume_dir, 'intermediate_models')

        if not os.path.exists(resume_model_directory):
            print(f"Resume directory {resume_model_directory} does not exist. Please check the path.")
            sys.exit(1)

        # Check to see if there are models that match the experiemnt name, env and seed, regardless of epochs
        matching_models = [f for f in os.listdir(resume_model_directory) if re.match(rf"{args.exp_name}_{args.env}_\d+_{args.seed}", f)]


        if matching_models:
            if len(matching_models) > 1:
                print(f"Multiple matching models found: {matching_models}.")
                sys.exit(1)
            print(f"Loading model from {matching_models[0]}")

            resume_model_dir = os.path.join(resume_model_directory, matching_models[0])

    exp_name_full = f"{args.exp_name}_{args.env}_{args.epochs*1000}_{args.seed}"


    output_dir = args.data_dir # Base data directory

    logger_kwargs = {
        'output_dir': output_dir, # Use potentially adjusted output_dir
        'output_fname': f"{exp_name_full}_progress.txt", # Log file name
        'exp_name': exp_name_full, # Experiment name used for subdirs
    }

    redq_sac(args.env, seed=args.seed, epochs=args.epochs,
             logger_kwargs=logger_kwargs, debug=args.debug,
             # Pass resume_path to the function
             resume_path=resume_model_dir if args.resume_dir else None)
