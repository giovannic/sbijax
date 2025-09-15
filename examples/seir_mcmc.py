"""
SEIR (Susceptible-Exposed-Infectious-Recovered) model implementation
using Flow Matching for Posterior Estimation.

This script implements a modern epidemiological model with:
- Both SFMPE and MCMC inference approaches
- Hydra configuration management

Updated to use latest package interfaces following hierarchical_brownian.py pattern.
This version performs estimation only and saves results to .npy files.
"""

import json
import logging
import time
from pathlib import Path
from jaxtyping import Array
import hydra
from omegaconf import DictConfig
from hydra.core.hydra_config import HydraConfig

from jax import numpy as jnp, random as jr, tree, jit
import tensorflow_probability.substrates.jax as tfp
from tensorflow_probability.substrates.jax import distributions as tfd
from tensorflow_probability.substrates.jax import bijectors as tfb

import arviz as az
import optax
from flax import nnx

from sfmpe.sfmpe import SFMPE
from sfmpe.bottom_up import train_bottom_up
from sfmpe.structured_cnf import StructuredCNF
from sfmpe.nn.transformer.transformer import Transformer
from sfmpe.pytree_bijector import (
    PyTreeBijector, 
    create_zscaling_bijector_tree
)

from seir_utils import (
    prior_fn, create_simulator_dist, create_simulator_fn,
    f_in_fn, flatten_theta_dict, apply_dequantization,
    create_flat_blockwise_bijector, reconstruct_theta_dict,
    f_in_fn_observed, _flatten, flatten_f_in,
    get_standard_bijector_specs, get_y_bijector_specs, create_pytree_bijectors,
    create_numpyro_seir_model, create_selective_prior_fn,
    create_selective_flat_bijector, flatten_selective_theta_dict,
    reconstruct_selective_theta_dict, create_selective_numpyro_seir_model
)


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

    # Check for selective inference configuration
    use_selective_inference = (
        hasattr(cfg, 'inference') and
        cfg.inference is not None and
        cfg.inference.sample_params is not None
    )

    if use_selective_inference:
        sample_params = cfg.inference.sample_params
        logger.info(f"Using selective inference: sampling {sample_params}")
        # All other parameters will be fixed at their true values
        all_params = ['beta_0', 'alpha', 'sigma', 'A', 'T_season', 'phi']
        fixed_param_names = [p for p in all_params if p not in sample_params]
        logger.info(f"Fixed parameters: {fixed_param_names}")
    else:
        logger.info("Using standard inference: sampling all parameters")

    key = jr.PRNGKey(cfg.seed)
    
    # Create functions
    simulator_dist = create_simulator_dist(n_timesteps, cfg.dt, cfg.population, cfg.I0_prop, n_warmup)
    simulator_fn = create_simulator_fn(simulator_dist)
    
    # Generate ground truth and observations
    theta_key, obs_key, f_in_key, key = jr.split(key, 4)
    
    theta_truth = prior_fn(n_sites).sample((1,), seed=theta_key)
    f_in = f_in_fn(n_obs, n_sites, n_timesteps).sample((1,), seed=f_in_key)
    y_observed = simulator_fn(obs_key, theta_truth, f_in)

    # Extract fixed parameter values for selective inference
    if use_selective_inference:
        fixed_params = {}
        for param_name in fixed_param_names:
            fixed_params[param_name] = theta_truth[param_name][0]  # Remove batch dim
        logger.info(f"Fixed parameter values: {tree.map(lambda x: jnp.squeeze(x), fixed_params)}")

    # Generate representative data for consistent Z-scaling across all bijectors
    repr_key, key = jr.split(key)
    repr_theta = prior_fn(n_sites).sample((1000,), seed=repr_key)

    # Generate prior samples for comparison plots
    prior_key, key = jr.split(key)
    if use_selective_inference:
        # For selective inference, generate samples only for sampled parameters
        selective_prior = create_selective_prior_fn(n_sites, sample_params, fixed_params)
        prior_samples_selective = selective_prior(n_sites).sample((cfg.n_prior_samples,), seed=prior_key)
        # Reconstruct full prior samples for plotting
        prior_samples_dict = reconstruct_selective_theta_dict(
            flatten_selective_theta_dict(prior_samples_selective, sample_params),
            sample_params, fixed_params, n_sites
        )
        prior_samples_flat = flatten_selective_theta_dict(prior_samples_dict, sample_params)[None, ...]
    else:
        prior_samples_dict = prior_fn(n_sites).sample((cfg.n_prior_samples,), seed=prior_key)
        prior_samples_flat = flatten_theta_dict(prior_samples_dict)[None, ...]  # Add chain dimension

    # Define bijector specifications for constrained -> unconstrained transformation
    bijector_specs = get_standard_bijector_specs()

    # Create MCMC bijector
    if use_selective_inference:
        flat_theta_bijector = create_selective_flat_bijector(repr_theta, bijector_specs, n_sites, sample_params)
    else:
        flat_theta_bijector = create_flat_blockwise_bijector(repr_theta, bijector_specs, n_sites)

    # Create proxy functions for MCMC sampling
    def flat_prior_fn(key: Array, n_samples: int) -> Array:
        """Prior function compatible with FMPE interface"""
        if use_selective_inference:
            selective_prior = create_selective_prior_fn(n_sites, sample_params, fixed_params)
            theta_samples = selective_prior(n_sites).sample((n_samples,), seed=key)
            return flatten_selective_theta_dict(theta_samples, sample_params)
        else:
            theta_samples = prior_fn(n_sites).sample((n_samples,), seed=key)
            return flatten_theta_dict(theta_samples)

    prior = prior_fn(n_sites)

    @jit
    def flat_simulator_log_prob(theta_flat: Array) -> Array:
        """Simulator function compatible with FMPE interface"""
        if use_selective_inference:
            # Reconstruct full theta from selective samples + fixed values
            theta_dict = reconstruct_selective_theta_dict(theta_flat, sample_params, fixed_params, n_sites)
            # Create selective prior for log_prob calculation
            selective_prior = create_selective_prior_fn(n_sites, sample_params, fixed_params)(n_sites)
            # Extract sampled parameters for prior calculation
            theta_selective = {k: v for k, v in theta_dict.items() if k in sample_params}
            prior_p = selective_prior.log_prob(theta_selective)
        else:
            # Reconstruct structured theta from flat representation
            theta_dict = reconstruct_theta_dict(theta_flat, n_sites)
            prior_p = prior.log_prob(theta_dict)

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

        if cfg.mcmc.sampler == "slice":
            init_state = flat_prior_fn(init_key, cfg.mcmc.n_chains)
            kernel = tfp.mcmc.TransformedTransitionKernel(
                inner_kernel=tfp.mcmc.SliceSampler(
                    target_log_prob_fn=flat_simulator_log_prob,
                    step_size=cfg.mcmc.step_size,
                    max_doublings=cfg.mcmc.max_doublings
                ),
                bijector=flat_theta_bijector
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

            if cfg.mcmc.use_numpyro_model:
                # Use NumPyro model approach
                logger.info(f"Using NumPyro model with {cfg.mcmc.sampler} sampler")

                # Create NumPyro model
                if use_selective_inference:
                    numpyro_model = create_selective_numpyro_seir_model(
                        simulator_fn, n_sites, f_in, sample_params, fixed_params
                    )
                else:
                    numpyro_model = create_numpyro_seir_model(simulator_fn, n_sites, f_in)

                if cfg.mcmc.sampler == "ess":
                    kernel = ESS(numpyro_model)
                    chain_method = "vectorized"
                else:
                    kernel = NUTS(
                        numpyro_model,
                        step_size=cfg.mcmc.step_size,
                        max_tree_depth=cfg.mcmc.max_tree_depth,
                        adapt_step_size=True,
                        forward_mode_differentiation=True
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

                if use_selective_inference:
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
                else:
                    # Reconstruct theta dictionary from NumPyro samples
                    theta_dict = {
                        'beta_0': samples['beta_0'][:, :, None, None],  # Add extra dims
                        'alpha': samples['alpha'][:, :, None, None],
                        'sigma': samples['sigma'][:, :, None, None],
                        'A': samples['A'][:, :, :, None],  # Site dimension preserved
                        'T_season': samples['T_season'][:, :, :, None],
                        'phi': samples['phi'][:, :, :, None]
                    }

                print(tree.map(lambda leaf: leaf.shape, theta_dict))

                # Convert to flat format for downstream analysis
                if use_selective_inference:
                    mcmc_posterior_samples = flatten_selective_theta_dict(theta_dict, sample_params)
                else:
                    mcmc_posterior_samples = flatten_theta_dict(theta_dict)

            else:
                # Use existing manual log_prob approach
                logger.info(f"Using manual log_prob with {cfg.mcmc.sampler} sampler")
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
                        forward_mode_differentiation=True
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
        repr_theta = prior_fn(n_sites).sample((1000,), seed=repr_key)
        
        # For representative data, use the same f_in for all samples
        repr_f_in = tree.map(lambda leaf: jnp.repeat(leaf, 1000, axis=0), f_in)
        repr_y_raw = simulator_fn(repr_key, repr_theta, repr_f_in)
        repr_y = apply_dequantization(repr_y_raw, deq_key)
        
        # Create Z-scaled bijector maps and PyTreeBijectors  
        theta_bijector_specs = get_standard_bijector_specs()
        y_bijector_specs = get_y_bijector_specs()
        sfmpe_theta_bijector, sfmpe_y_bijector = create_pytree_bijectors(repr_theta, repr_y, theta_bijector_specs, y_bijector_specs)
        
        # Transform observations to unconstrained space
        y_unconstrained = sfmpe_y_bijector.forward(y_processed)
        
        # Create wrapped functions for train_bottom_up
        def wrapped_prior_fn(n):
            """Prior function that returns TransformedDistribution."""
            base_prior = prior_fn(n)
            return tfd.TransformedDistribution(
                base_prior, 
                sfmpe_theta_bijector,
                name="transformed_prior"
            )
        
        def p_local(g, n):
            """Local prior distribution for site-specific parameters (independent of global)."""
            n_sims = g['beta_0'].shape[0]
            t_season_spread = 1./50.
            return tfd.JointDistributionNamed(
                dict(
                    # Site-specific seasonal parameters  
                    A = tfd.Uniform(jnp.zeros((n_sims, n, 1)), jnp.ones((n_sims, n, 1))),
                    T_season = tfd.Gamma(
                        jnp.full((n_sims, n, 1), 365.0 * t_season_spread), 
                        jnp.full((n_sims, n, 1), t_season_spread)
                    ),
                    phi = tfd.Uniform(
                        jnp.zeros((n_sims, n, 1)), 
                        jnp.full((n_sims, n, 1), 2*jnp.pi)
                    )
                ),
                batch_ndims=1,
            )
        
        def wrapped_p_local(g, n):
            """Local prior function that returns TransformedDistribution."""
            base_local = p_local(g, n)
            return tfd.TransformedDistribution(
                base_local,
                sfmpe_theta_bijector,  # Same bijector as wrapped_prior_fn
                name="transformed_local"
            )
        
        def wrapped_simulator_fn(seed, theta, f_in_sample):
            """Simulator function that handles bijector transformations."""
            # Transform parameters back to constrained space
            theta_constrained = sfmpe_theta_bijector.inverse(theta)
            
            # Apply original simulator
            y_constrained = simulator_fn(seed, theta_constrained, f_in_sample)
            y_deq = apply_dequantization(y_constrained, seed)
            
            # Transform outputs to unconstrained space
            return sfmpe_y_bijector.forward(y_deq)
        
        # Independence structure for structured inference
        independence = {
            'local': ['obs'],  # Observations independent across time/sites
            'cross': [],
            'cross_local': [  # Site-specific parameters connect to their observations
                ('A', 'obs', (0, 0)),
                ('T_season', 'obs', (0, 0)),
                ('phi', 'obs', (0, 0))
            ]
        }
        
        # SFMPE Neural Network Setup
        rngs = nnx.Rngs(key)
        transformer_config = {
            'latent_dim': cfg.sfmpe.transformer.latent_dim,
            'label_dim': cfg.sfmpe.transformer.label_dim,
            'index_out_dim': cfg.sfmpe.transformer.index_out_dim,
            'n_encoder': cfg.sfmpe.transformer.n_encoder,
            'n_decoder': cfg.sfmpe.transformer.n_decoder,
            'n_heads': cfg.sfmpe.transformer.n_heads,
            'n_ff': cfg.sfmpe.transformer.n_ff,
            'dropout': cfg.sfmpe.transformer.dropout,
            'activation': nnx.relu,
        }

        nn = Transformer(
            transformer_config,
            value_dim=1,
            n_labels=7,  # 3 global + 3 site-specific parameters + obs
            index_dim=1,  # Temporal indexing
            rngs=rngs
        )

        model = StructuredCNF(nn, rngs=rngs)
        estim = SFMPE(model, rngs=rngs)

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
        
        labels, slices, masks = train_bottom_up(
            train_key,
            estim,
            wrapped_prior_fn,
            wrapped_p_local,
            wrapped_simulator_fn,
            ['beta_0', 'alpha', 'sigma'],  # Global parameters
            ['A', 'T_season', 'phi'],  # Local parameters
            n_sites,
            n_rounds,
            n_simulations,
            n_epochs,
            y_unconstrained,  # Use unconstrained data
            independence,
            optimiser=optax.adam(cfg.training.learning_rate),
            batch_size=int(n_simulations * cfg.training.batch_size_fraction),
            f_in=f_in_fn_train,
            f_in_args=f_in_args,
            f_in_args_global=f_in_args_global,
            f_in_target=f_in
        )
        logger.info(f"SFMPE bottom-up training completed in {time.time() - start_time:.2f} seconds")

        # Sample SFMPE posterior
        logger.info("Sampling SFMPE posterior")
        start_time = time.time()
        
        # Create flattened f_in index for posterior sampling
        f_in_flattened = flatten_f_in(f_in)
        posterior = estim.sample_posterior(
            _flatten(y_processed)[..., None],
            labels,
            slices,
            masks=masks,
            n_samples=n_post_samples,
            index=f_in_flattened
        )

        # Transform posterior samples back into constrained space
        posterior = sfmpe_theta_bijector.inverse(posterior)

        # Convert SFMPE posterior to the same format as MCMC for downstream analysis
        if use_selective_inference:
            mcmc_posterior_samples = flatten_selective_theta_dict(posterior, sample_params)[None, ...]
        else:
            mcmc_posterior_samples = flatten_theta_dict(posterior)[None, ...]
        
        logger.info(f'SFMPE posterior mean: {jnp.mean(mcmc_posterior_samples, axis=(0, 1))}')
        logger.info(f"SFMPE posterior sampling completed in {time.time() - start_time:.2f} seconds")
        
    else:
        raise ValueError(f"Unknown method: {cfg.method}. Choose 'MCMC' or 'SFMPE'.")

    logger.info(f'Analysing MCMC')
    start_time = time.time()
    logger.info(f"Converting MCMC posterior to az format")

    if use_selective_inference:
        # Reconstruct full theta dict from selective samples + fixed values
        post_dict = reconstruct_selective_theta_dict(
            mcmc_posterior_samples,
            sample_params,
            fixed_params,
            n_sites
        )
    else:
        post_dict = reconstruct_theta_dict(
            mcmc_posterior_samples,
            n_sites
        )
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
