"""
SEIR (Susceptible-Exposed-Infectious-Recovered) model implementation
using Flow Matching for Posterior Estimation.

This script implements a modern epidemiological model with:
- Both SFMPE and MCMC inference approaches
- Hydra configuration management

Updated to use latest package interfaces following hierarchical_brownian.py pattern.
This version performs estimation only and saves results to .npy files.
"""

import jax
print(f"The jax default backend is: {jax.default_backend()}")

import json
import logging
import time
from pathlib import Path
from jaxtyping import Array, PyTree
from typing import Callable

import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from jax import numpy as jnp, random as jr, tree, jit
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd

import arviz as az
import optax
from flax import nnx


from tfmpe.estimators.tfmpe import TFMPE, NormalDistribution
from tfmpe.estimators.training import fit_bottom_up
from tfmpe.preprocessing.tokens import Tokens
from tfmpe.preprocessing.utils import Independence, Labeller
from tfmpe.nn.transformer import Transformer, TransformerConfig
import diffrax

from seir_utils import (
    prior_fn, create_simulator_dist, create_simulator_fn,
    f_in_fn, apply_dequantization,
    f_in_fn_observed, _flatten, flatten_f_in,
    get_standard_bijector_specs, get_y_bijector_specs, create_pytree_bijectors,
    create_selective_prior_fn,
    create_selective_flat_bijector, flatten_selective_theta_dict,
    reconstruct_selective_theta_dict, create_selective_numpyro_seir_model,
    create_selective_sfmpe_functions, sbc_plot
)
import matplotlib.pyplot as plt


def run(cfg: DictConfig) -> None:
    """Main execution function."""
    logger = logging.getLogger(__name__)
    logger.info(f"Running SEIR with n_simulations={cfg.n_simulations}, "
                f"n_rounds={cfg.n_rounds}, n_epochs={cfg.n_epochs}")

    # Extract parameters
    n_timesteps = cfg.n_timesteps
    n_obs = cfg.n_obs
    n_sites = cfg.n_sites
    n_warmup = cfg.n_warmup
    n_simulations = cfg.n_simulations
    n_rounds = cfg.n_rounds
    n_epochs = cfg.n_epochs
    n_post_samples = cfg.n_post_samples

    # Set up parameters for unified selective approach
    # Full inference is just selective inference with all parameters sampled
    all_params = ['beta_0', 'alpha', 'sigma', 'A', 'T_season', 'phi']

    if (hasattr(cfg, 'inference') and
        cfg.inference is not None and
        cfg.inference.sample_params is not None):
        sample_params = cfg.inference.sample_params
        logger.info(f"Using selective inference: sampling {sample_params}")
    else:
        sample_params = all_params
        logger.info("Using full inference: sampling all parameters")

    fixed_param_names = [p for p in all_params if p not in sample_params]
    if fixed_param_names:
        logger.info(f"Fixed parameters: {fixed_param_names}")
    else:
        logger.info("No fixed parameters")

    key = jr.PRNGKey(cfg.seed)
    
    # Create functions
    simulator_dist = create_simulator_dist(n_timesteps, cfg.dt, cfg.population, cfg.I0_prop, n_warmup)
    simulator_fn = create_simulator_fn(simulator_dist)
    
    # Generate ground truth and observations
    theta_key, obs_key, f_in_key, key = jr.split(key, 4)
    
    theta_truth = prior_fn(n_sites).sample((1,), seed=theta_key)

    # For selective inference, broadcast fixed local parameters to be identical across sites
    for param_name in fixed_param_names:
        if param_name in ['A', 'T_season', 'phi']:
            # Take the first site's value and broadcast it across all sites
            single_value = theta_truth[param_name][0, 0:1]  # Shape: (1, 1)
            broadcasted_value = jnp.broadcast_to(single_value, (n_sites, 1))
            theta_truth[param_name] = theta_truth[param_name].at[0].set(broadcasted_value)

    f_in = f_in_fn(f_in_key, 1, n_obs, n_sites, n_timesteps)
    y_observed = simulator_fn(obs_key, theta_truth, f_in)

    # Extract fixed parameter values
    fixed_params = {}
    for param_name in fixed_param_names:
        fixed_params[param_name] = theta_truth[param_name][0]  # Remove batch dim
    if fixed_params:
        logger.info(f"Fixed parameter values: {tree.map(lambda x: jnp.squeeze(x), fixed_params)}")

    # Generate representative data for consistent Z-scaling across all bijectors
    repr_key, key = jr.split(key)
    repr_theta = prior_fn(n_sites).sample((1000,), seed=repr_key)

    # Generate prior samples for comparison plots
    prior_key, key = jr.split(key)
    selective_prior = create_selective_prior_fn(n_sites, sample_params, fixed_params)
    prior_samples_selective = selective_prior(n_sites).sample((cfg.n_prior_samples,), seed=prior_key)
    prior_samples_flat = flatten_selective_theta_dict(prior_samples_selective, sample_params)[None, ...]

    # Define bijector specifications for constrained -> unconstrained transformation
    bijector_specs = get_standard_bijector_specs()

    # Create MCMC bijector (always use selective approach)
    flat_theta_bijector = create_selective_flat_bijector(repr_theta, bijector_specs, n_sites, sample_params)

    # Create proxy functions for MCMC sampling
    def flat_prior_fn(key: Array, n_samples: int) -> Array:
        """Prior function compatible with FMPE interface"""
        selective_prior = create_selective_prior_fn(n_sites, sample_params, fixed_params)
        theta_samples = selective_prior(n_sites).sample((n_samples,), seed=key)
        return flatten_selective_theta_dict(theta_samples, sample_params)

    @jit
    def flat_simulator_log_prob(theta_flat: Array) -> Array:
        """Simulator function compatible with FMPE interface"""
        # Reconstruct full theta from selective samples + fixed values
        theta_dict = reconstruct_selective_theta_dict(theta_flat, sample_params, fixed_params, n_sites)
        # Create selective prior for log_prob calculation
        selective_prior = create_selective_prior_fn(n_sites, sample_params, fixed_params)(n_sites)
        # Extract sampled parameters for prior calculation
        theta_selective = {k: v for k, v in theta_dict.items() if k in sample_params}
        prior_p = selective_prior.log_prob(theta_selective)

        # Run simulator
        n_simulations = theta_flat.shape[0]
        f_in_matched = tree.map(
            lambda leaf: jnp.repeat(leaf, n_simulations, axis=0),
            f_in
        )
        sim_dist = simulator_dist(theta_dict, f_in_matched)
        return jnp.sum(prior_p, axis=1) + jnp.sum(sim_dist.log_prob(y_observed), axis=(1, 2))

    if cfg.method == "MCMC":
        # Train using round-based approach
        logger.info("Starting MCMC sampling")
        start_time = time.time()

        # Sample from MCMC
        n_burnin = cfg.n_simulations - cfg.n_post_samples
        sample_key, init_key, key = jr.split(key, 3)

        if cfg.mcmc.sampler == "nuts_tfp":
            if cfg.mcmc.init_to_truth:
                flat_truth = flatten_selective_theta_dict(theta_truth, sample_params)
                init_state = jnp.broadcast_to(
                    flat_truth,
                    (cfg.mcmc.n_chains, flat_truth.shape[-1])
                )
            else:
                init_state = flat_prior_fn(init_key, cfg.mcmc.n_chains)

            kernel = tfp.mcmc.NoUTurnSampler(
                target_log_prob_fn=flat_simulator_log_prob,
                step_size=cfg.mcmc.step_size,
                max_tree_depth=2
            )

            mcmc_posterior_samples = tfp.mcmc.sample_chain(
                num_results=n_post_samples,
                num_burnin_steps=n_burnin,
                current_state=init_state,
                kernel=kernel,
                seed=sample_key
            )
            mcmc_posterior_samples = mcmc_posterior_samples.all_states
            # change axes so that it's [chain, sample, param]
            mcmc_posterior_samples = jnp.swapaxes(
                mcmc_posterior_samples,
                0,
                1
            )
        elif cfg.mcmc.sampler in ["nuts", "ess"]:
            from numpyro.infer import MCMC, NUTS
            from numpyro.infer.ensemble import ESS
            from numpyro.infer.initialization import init_to_value, init_to_uniform

            if cfg.mcmc.use_numpyro_model:
                # Use NumPyro model approach
                logger.info(f"Using NumPyro model with {cfg.mcmc.sampler} sampler")

                # Create NumPyro model (always use selective approach)
                numpyro_model = create_selective_numpyro_seir_model(
                    simulator_fn, n_sites, f_in, sample_params, fixed_params
                )

                if cfg.mcmc.init_to_truth:
                    truth_for_sampling = {
                        name: value[0, ..., 0]
                        if name in {'A', 'T_season', 'phi'} else value[0, 0, 0]
                        for name, value in theta_truth.items()
                    }
                    init_strategy = init_to_value(
                        values = truth_for_sampling
                    )
                else:
                    init_strategy = init_to_uniform

                if cfg.mcmc.sampler == "ess":
                    kernel = ESS(
                        numpyro_model,
                        randomize_split=True,
                        moves={
                            AIES.DEMove() : 0.5,
                            AIES.StretchMove() : 0.5
                        },
                        init_strategy=init_strategy)
                    chain_method = "vectorized"
                else:
                    kernel = NUTS(
                        numpyro_model,
                        init_strategy=init_strategy,
                        step_size=cfg.mcmc.step_size,
                        max_tree_depth=cfg.mcmc.max_tree_depth,
                        adapt_step_size=True
                    )
                    chain_method = "parallel"

                mcmc = MCMC(
                    kernel,
                    num_warmup=n_burnin,
                    num_samples=n_post_samples,
                    chain_method=chain_method,
                    num_chains=cfg.mcmc.n_chains,
                    jit_model_args=True
                )
                mcmc.run(sample_key, y_observed=y_observed)

                # Extract samples and convert to expected format
                samples = mcmc.get_samples(group_by_chain=True)

                # Reconstruct theta dictionary from selective NumPyro samples
                theta_dict = {}
                for param_name in sample_params:
                    if param_name in ['beta_0', 'alpha', 'sigma']:
                        theta_dict[param_name] = samples[param_name][:, :, None, None]
                    else:
                        theta_dict[param_name] = samples[param_name][:, :, :, None]

                # Add fixed parameters
                for param_name in fixed_param_names:
                    fixed_val = fixed_params[param_name]
                    batch_shape = samples[sample_params[0]].shape[:2]  # [n_chains, n_samples]
                    if param_name in ['beta_0', 'alpha', 'sigma']:
                        theta_dict[param_name] = jnp.broadcast_to(
                            fixed_val[None, None, :, :],
                            batch_shape + (1, 1)
                        )
                    else:
                        theta_dict[param_name] = jnp.broadcast_to(
                            fixed_val[None, None, :, :],
                            batch_shape + (n_sites, 1)
                        )

                # Convert to flat format for downstream analysis
                mcmc_posterior_samples = flatten_selective_theta_dict(theta_dict, sample_params)

            else:
                # Use existing manual log_prob approach
                logger.info(f"Using manual log_prob with {cfg.mcmc.sampler} sampler")
                if cfg.mcmc.init_to_truth:
                    flat_truth = flatten_selective_theta_dict(theta_truth, sample_params)
                    init_state = jnp.broadcast_to(
                        flat_truth,
                        (cfg.mcmc.n_chains, flat_truth.shape[-1])
                    )
                else:
                    init_state = flat_prior_fn(init_key, cfg.mcmc.n_chains)

                def transformed_log_prob(theta: Array) -> Array:
                    batched_theta = theta[None, ...]
                    unconstrained_theta = flat_theta_bijector.forward(batched_theta)
                    log_prob = flat_simulator_log_prob(unconstrained_theta)[0]
                    det = flat_theta_bijector.forward_log_det_jacobian(
                        batched_theta
                    )[0]
                    return log_prob + det

                if cfg.mcmc.sampler == "ess":
                    kernel = ESS(
                        potential_fn=transformed_log_prob
                    )
                    chain_method = "vectorized"
                else:
                    kernel = NUTS(
                        potential_fn=transformed_log_prob,
                        step_size=cfg.mcmc.step_size,
                        max_tree_depth=cfg.mcmc.max_tree_depth,
                        adapt_step_size=True,
                    )
                    chain_method = "parallel"

                mcmc = MCMC(
                    kernel,
                    num_warmup=n_burnin,
                    num_samples=n_post_samples,
                    chain_method=chain_method,
                    num_chains=cfg.mcmc.n_chains,
                    jit_model_args=True
                )
                mcmc.run(sample_key, init_params=flat_theta_bijector.forward(init_state))
                unconstrained_samples = mcmc.get_samples(group_by_chain=True)
                mcmc_posterior_samples = flat_theta_bijector.inverse(unconstrained_samples)
        else:
            raise ValueError(f"Unknown MCMC sampler: {cfg.mcmc.sampler}")


        logger.info(f'MCMC posterior mean: {jnp.mean(mcmc_posterior_samples, axis=(0, 1))}')
        logger.info(f"MCMC posterior sampling completed in {time.time() - start_time:.2f} seconds")

    elif cfg.method == "SFMPE":
        # SFMPE implementation
        logger.info("Starting SFMPE training")
        start_time = time.time()
        
        # Apply dequantization to observed data
        deq_key, key = jr.split(key)
        y_processed = apply_dequantization(y_observed, deq_key)
        
        # Generate representative data for consistent Z-scaling across all bijectors
        repr_key, key = jr.split(key)
        # Always use selective functions for representative data generation
        selective_prior_fn, selective_local_fn, wrapped_simulator_fn, global_names, local_names = create_selective_sfmpe_functions(
            n_sites, sample_params, fixed_params, simulator_fn
        )
        # Sample only the selected parameters for representative data
        repr_theta = selective_prior_fn(n_sites).sample((1000,), seed=repr_key)

        # For representative data, use the same f_in for all samples
        repr_f_in = tree.map(lambda leaf: jnp.repeat(leaf, 1000, axis=0), f_in)
        repr_y_raw = wrapped_simulator_fn(repr_key, repr_theta, repr_f_in)
        repr_y = apply_dequantization(repr_y_raw, deq_key)

        # Create Z-scaled bijector maps and PyTreeBijectors
        # Filter bijector specs to only include sampled parameters
        theta_bijector_specs = get_standard_bijector_specs()
        selective_theta_specs = {k: v for k, v in theta_bijector_specs.items() if k in sample_params}
        y_bijector_specs = get_y_bijector_specs()
        sfmpe_theta_bijector, sfmpe_y_bijector = create_pytree_bijectors(repr_theta, repr_y, selective_theta_specs, y_bijector_specs)
        
        # Transform observations to unconstrained space
        y_unconstrained = sfmpe_y_bijector.forward(y_processed)
        
        # Create wrapped functions for train_bottom_up
        def wrapped_prior_fn(rng, n, n_samples, f_in):
            """Prior function that returns TransformedDistribution."""
            base_prior = selective_prior_fn(n)
            return tfd.TransformedDistribution(
                base_prior,
                sfmpe_theta_bijector,
                name="transformed_prior"
            ).sample(n_samples, seed=rng)

        def wrapped_p_local(rng, g, n, f_in):
            """Local prior function that returns TransformedDistribution."""
            base_local = selective_local_fn(g, n)
            samples = tfd.TransformedDistribution(
                base_local,
                sfmpe_theta_bijector,
                name="transformed_local"
                ).sample(1, seed=rng)
            return {k: v[0] for k, v in samples.items()}
        
        def wrapped_simulator_fn_for_training(seed, theta, n, f_in_sample):
            """Simulator function that handles bijector transformations."""
            # Transform parameters back to constrained space
            theta_constrained = sfmpe_theta_bijector.inverse(theta)

            # Apply wrapped simulator (handles parameter reconstruction if needed)
            y_constrained = wrapped_simulator_fn(seed, theta_constrained, f_in_sample)
            y_deq = apply_dequantization(y_constrained, seed)

            # Transform outputs to unconstrained space
            return sfmpe_y_bijector.forward(y_deq)
     
        independence = Independence()

        # SFMPE Neural Network Setup (dynamic n_labels) using estim_key
        param_key, dropout_key = jr.split(key)
        rngs = nnx.Rngs(params=param_key, dropout=dropout_key)
        n_labels = len(global_names) + len(local_names) + 1  # sampled parameters + obs
        logger.info(f"Using {n_labels} labels: {len(global_names)} global + {len(local_names)} local + 1 obs")

        transformer_config = TransformerConfig(
            latent_dim = cfg.sfmpe.transformer.latent_dim,
            n_encoder = cfg.sfmpe.transformer.n_encoder,
            n_decoder = cfg.sfmpe.transformer.n_decoder,
            n_heads = cfg.sfmpe.transformer.n_heads,
            n_ff = cfg.sfmpe.transformer.n_ff,
            label_dim = cfg.sfmpe.transformer.label_dim,
            dropout = cfg.sfmpe.transformer.dropout,
            index_out_dim = cfg.sfmpe.transformer.index_out_dim,
        )

        labeller = Labeller.for_keys(list(repr_theta.keys()) + ['obs'])

        repr_tokens = Tokens.from_pytree(
            repr_theta,
            independence=independence,
            labeller=labeller,
            functional_inputs=repr_f_in
        )

        base_dist = NormalDistribution(rngs=rngs)

        nn = Transformer(
            config=transformer_config,
            tokens=repr_tokens,
            rngs=rngs
        )

        estim = TFMPE(
            vf_network=nn,
            base_dist=base_dist,
            solver=diffrax.Dopri5(),
        )


        # Training
        train_key, key = jr.split(key)
        logger.info("Starting SFMPE bottom-up training")

        # Set up f_in function arguments based on configuration
        if cfg.f_in_sample == 'observed':
            f_in_fn_train = f_in_fn_observed
            f_in_args = (n_obs, 1, f_in)
            f_in_args_global = (n_obs, n_sites, f_in)
        elif cfg.f_in_sample == 'prior':
            f_in_fn_train = f_in_fn
            f_in_args = (n_obs, 1, n_timesteps)
            f_in_args_global = (n_obs, n_sites, n_timesteps)
        else:
            raise ValueError(f"Invalid f_in_sample: {cfg.f_in_sample}")

        start_time = time.time()
        estim, losses = fit_bottom_up(
            tfmpe=estim,
            y_obs=y_unconstrained,  # Use unconstrained data
            simulator_fn=wrapped_simulator_fn_for_training,  # Use training-specific wrapped simulator
            prior_fn=wrapped_prior_fn,
            local_fn=wrapped_p_local,
            global_names=global_names,  # Dynamic global parameters
            n_groups=n_sites,
            n_rounds=n_rounds,
            n_samples_per_round=n_simulations,
            n_val_samples=100,
            opt=nnx.Optimizer(
                estim,
                optax.adam(cfg.training.learning_rate),
                wrt=nnx.Param
            ),
            n_iter_per_round=n_epochs,
            batch_size=100,
            rng=train_key,
            independence=independence,
            labeller=labeller,
            f_in_fn = f_in_fn_train,
            f_in_args = f_in_args,
            f_in_args_global = f_in_args_global,
        )
        logger.info(f"SFMPE bottom-up training completed in {time.time() - start_time:.2f} seconds")

        # Sample SFMPE posterior
        logger.info("Sampling SFMPE posterior")
        start_time = time.time()

        def f_in_for_samples(n_samples):
            return tree.map(
                lambda leaf: jnp.broadcast_to(
                    leaf,
                    (n_samples,) + leaf.shape[1:]
                ),
                f_in
            )

        param_tokens = Tokens.from_pytree(
            tree.map(
                lambda leaf: jnp.zeros((n_post_samples,) + leaf.shape[1:]),
                repr_theta
            ),
            independence=independence,
            labeller=labeller,
            functional_inputs=f_in_for_samples(n_post_samples),
        )
        context_tokens = Tokens.from_pytree(
            tree.map(
                lambda leaf: jnp.broadcast_to(leaf, (n_post_samples,) + leaf.shape[1:]),
                y_unconstrained
            ),
            independence=independence,
            labeller=labeller,
            functional_inputs=f_in_for_samples(n_post_samples)
        )

        posterior_tokens = estim.sample_posterior(
            context=context_tokens,
            params=param_tokens
        )

        posterior_unconstrained = posterior_tokens.decode()

        # Transform to constrained space for true posterior evaluation
        posterior = sfmpe_theta_bijector.inverse(posterior_unconstrained)

        # Convert SFMPE posterior to the same format as MCMC for downstream analysis
        mcmc_posterior_samples = flatten_selective_theta_dict(posterior, sample_params)[None, ...]
        
        def compute_sfmpe_ranks_batched(
            key: jnp.ndarray,
            n_tests: int,
            n_posterior_samples: int,
        ) -> Array:
            """
            Compute SBC ranks for multiple test cases in a single batch.

            Parameters
            ----------
            key : jnp.ndarray
                PRNG key
            n_tests : int
                Number of SBC test cases
            n_posterior_samples : int
                Number of posterior samples per test case

            Returns
            -------
            Array of shape (n_tests, param_dim) containing ranks
            """
            prior_key, sim_key = jr.split(key)

            # 1. Generate n_tests prior samples (batched)
            priors = selective_prior_fn(n_sites).sample((n_tests,), seed=prior_key)
            # Shape: {param: (n_tests, ...)} for each param

            # 2. Generate n_tests observations using vmap
            def simulate_single(sim_key: Array, prior_sample: PyTree) -> PyTree:
                # prior_sample: {param: (...)} without batch dim
                # Add batch dim back for simulator
                prior_batched = tree.map(lambda x: x[None, ...], prior_sample)
                y = wrapped_simulator_fn(sim_key, prior_batched, f_in)
                return apply_dequantization(y, sim_key)

            sim_keys = jr.split(sim_key, n_tests)
            # vmap over test cases
            observations = jax.vmap(simulate_single)(
                sim_keys,
                priors  # Tree of {param: (n_tests, ...)}
            )
            # Shape: (n_tests, 1, n_obs, n_sites, n_timesteps)

            # Transform to unconstrained space
            observations_unconstrained = sfmpe_y_bijector.forward(observations)

            # 3. Prepare for sample_posterior - replicate each observation n times
            total_samples = n_tests * n_posterior_samples
            context_repeated = tree.map(
                lambda x: jnp.repeat(x, n_posterior_samples, axis=0),
                observations_unconstrained
            )
            # Shape: (n_tests * n, 1, n_obs, n_sites, n_timesteps)

            # Create f_in for all samples
            f_in_all = f_in_for_samples(total_samples)

            # Create context tokens
            context_tokens = Tokens.from_pytree(
                context_repeated,
                independence=independence,
                labeller=labeller,
                functional_inputs=f_in_all
            )

            # Create param tokens (template for posterior samples)
            param_tokens = Tokens.from_pytree(
                {k: jnp.zeros((total_samples,) + v.shape[1:])
                 for k, v in repr_theta.items()},
                independence=independence,
                labeller=labeller,
                functional_inputs=f_in_all
            )

            # 4. Single call to sample_posterior for all samples
            posterior_tokens = estim.sample_posterior_batched(
                context=context_tokens,
                params=param_tokens,
                batch_size=1000
            )
            posterior_unconstrained = posterior_tokens.decode()

            # Transform to constrained space
            posterior_constrained = sfmpe_theta_bijector.inverse(posterior_unconstrained)

            # 5. Reshape and compute ranks
            # Flatten posterior to (n_tests * n, param_dim)
            posterior_flat = flatten_selective_theta_dict(posterior_constrained, sample_params)
            # Reshape to (n_tests, n, param_dim)
            param_dim = posterior_flat.shape[-1]
            posterior_reshaped = posterior_flat.reshape(n_tests, n_posterior_samples, param_dim)

            # Flatten priors to (n_tests, param_dim)
            priors_flat = flatten_selective_theta_dict(priors, sample_params)

            # Compute ranks: count posterior samples < prior value
            # priors_flat[:, None, :] broadcasts to (n_tests, 1, param_dim)
            ranks = jnp.sum(posterior_reshaped < priors_flat[:, None, :], axis=1)
            # Shape: (n_tests, param_dim)

            return ranks

        max_rank = 100
        n_tests = 100
        ranks = compute_sfmpe_ranks_batched(key, n_tests, max_rank)
        # plot SBC rank plot
        fig = sbc_plot(ranks, max_rank, sample_params, n_sites)

        # save figure to sbc.png
        sbc_out_dir = Path(HydraConfig.get().runtime.output_dir)
        fig.savefig(sbc_out_dir / "sbc.png", dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"SBC plot saved to {sbc_out_dir / 'sbc.png'}")

        logger.info(f'SFMPE posterior mean: {jnp.mean(mcmc_posterior_samples, axis=(0, 1))}')
        logger.info(f"SFMPE posterior sampling completed in {time.time() - start_time:.2f} seconds")
    else:
        raise ValueError(f"Unknown method: {cfg.method}. Choose 'MCMC' or 'SFMPE'.")

    logger.info(f'Analysing MCMC')
    start_time = time.time()
    logger.info(f"Converting MCMC posterior to az format")

    # Reconstruct only sampled parameters for ArviZ
    # First reconstruct with fixed params to get proper shapes, then filter
    post_dict_full = reconstruct_selective_theta_dict(
        mcmc_posterior_samples,
        sample_params,
        fixed_params,
        n_sites
    )
    post_dict = {k: v for k, v in post_dict_full.items() if k in sample_params}
    posterior = az.from_dict(posterior=post_dict)
    logger.info(f"Summarising MCMC posterior")
    print(az.summary(posterior))
    logger.info(f"MCMC summarisation completed in {time.time() - start_time:.2f} seconds")

    # Use Hydra's output directory
    hydra_cfg = HydraConfig.get()
    out_dir = Path(hydra_cfg.runtime.output_dir)
    
    # Save estimation results as .npy files for later visualization
    logger.info("Saving results to .npy files")
    jnp.save(out_dir / "theta_truth.npy", theta_truth)
    jnp.save(out_dir / "y_observed.npy", y_observed)
    jnp.save(out_dir / "f_in.npy", f_in)
    jnp.save(out_dir / "mcmc_posterior_samples.npy", mcmc_posterior_samples)
    jnp.save(out_dir / "prior_samples.npy", prior_samples_flat)
    
    # Save configuration parameters needed for plotting
    plot_config = {
        'n_sites': n_sites,
        'n_timesteps': n_timesteps,
        'n_warmup': n_warmup,
        'population': cfg.population,
        'I0_prop': cfg.I0_prop,
        'dt': cfg.dt
    }

    # Save as JSON for easy loading
    with open(out_dir / "plot_config.json", 'w') as f:
        json.dump(plot_config, f, indent=2)

    # Always save selective inference configuration
    selective_config = {
        'sample_params': list(sample_params),  # Convert from ListConfig to list
        'fixed_params': tree.map(lambda x: x.tolist(), fixed_params)
    }
    with open(out_dir / "selective_inference_config.json", 'w') as f:
        json.dump(selective_config, f, indent=2)
    
    logger.info("SEIR MCMC estimation completed successfully!")

@hydra.main(version_base=None, config_path="conf", config_name="seir_mcmc_config")
def main(cfg: DictConfig) -> None:
    """Main function with Hydra configuration management."""
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting SEIR experiment")
    
    # Run the experiment
    run(cfg)

if __name__ == "__main__":
    main()
