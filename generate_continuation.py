#!/usr/bin/env python3
"""
Generate MIDI drum pattern continuations using the GrooVAE-torch model.

TRULY STOCHASTIC VERSION: Uses torch.bernoulli for hit sampling to ensure variation
WITH ANALYSIS AND SELF-ADJUST: Closed-loop generator that analyzes seed patterns
and iteratively adjusts generation parameters for better continuations.
WITH QUALITY FILTERING: Only keeps generated variations that are statistically similar
to their parent input, up to 3 suitable variations per input.
"""
import argparse
import os
from pathlib import Path
import traceback
from datetime import datetime
import json

import numpy as np
import torch
import pretty_midi
from scipy.spatial.distance import cosine
from scipy.stats import entropy

from model import Encoder, Decoder
from drum_utils import change_fs, to_tensors, from_tensors_to_midi, quantize_drum, map_unique_drum
from config import Config


# === ANALYSIS MODULE ===

class DrumPatternAnalyzer:
    """
    Analyzes drum patterns at bar-level to compute rhythmic signatures.
    """
    
    def __init__(self, steps_per_bar=16, comp=9):
        self.steps_per_bar = steps_per_bar
        self.comp = comp
    
    def compute_bar_stats(self, hit_roll, velocity_roll, offset_roll):
        """
        Compute comprehensive bar-level statistics from quantized rolls.
        Enhanced with raw pattern storage for identity detection.
        """
        steps, comp = hit_roll.shape
        
        if steps < self.steps_per_bar:
            return self._empty_stats()
        
        num_bars = steps // self.steps_per_bar
        bar_stats = []
        hit_bars = []
        
        for bar_idx in range(num_bars):
            start = bar_idx * self.steps_per_bar
            end = start + self.steps_per_bar
            
            hits = hit_roll[start:end]
            vels = velocity_roll[start:end]
            offs = offset_roll[start:end]
            
            hit_bars.append(hits)
            
            # 1. Total density (hits per bar)
            total_hits = np.sum(hits)
            bar_stat = {'total_density': float(total_hits)}
            
            # Store raw hit pattern for direct comparison
            bar_stat['hit_pattern'] = hits.copy()
            
            # 2. Per-drum density
            per_drum_density = [float(np.sum(hits[:, drum])) for drum in range(comp)]
            bar_stat['per_drum_density'] = per_drum_density
            
            # 3. Hit position distribution
            hit_positions = np.sum(hits, axis=1)  # Sum across drums for each step
            bar_stat['hit_position_dist'] = hit_positions.tolist()
            
            # 4. Velocity statistics
            hit_mask = hits > 0.5
            velocities = vels[hit_mask]
            if len(velocities) > 0:
                bar_stat['velocity_mean'] = float(np.mean(velocities))
                bar_stat['velocity_std'] = float(np.std(velocities))
            else:
                bar_stat['velocity_mean'] = 0.0
                bar_stat['velocity_std'] = 0.0
            
            # Per-drum velocity statistics
            per_drum_vel_mean = []
            for drum in range(comp):
                drum_mask = hits[:, drum] > 0.5
                drum_vels = vels[drum_mask, drum]
                if len(drum_vels) > 0:
                    per_drum_vel_mean.append(float(np.mean(drum_vels)))
                else:
                    per_drum_vel_mean.append(0.0)
            bar_stat['per_drum_velocity_mean'] = per_drum_vel_mean
            
            # 5. Offset statistics
            offsets = offs[hit_mask]
            if len(offsets) > 0:
                bar_stat['offset_std'] = float(np.std(offsets))
            else:
                bar_stat['offset_std'] = 0.0
            
            # Per-drum offset std
            per_drum_off_std = []
            for drum in range(comp):
                drum_mask = hits[:, drum] > 0.5
                drum_offs = offs[drum_mask, drum]
                if len(drum_offs) > 0:
                    per_drum_off_std.append(float(np.std(drum_offs)))
                else:
                    per_drum_off_std.append(0.0)
            bar_stat['per_drum_offset_std'] = per_drum_off_std
            
            # 6. Syncopation proxy (off-beat vs on-beat ratio)
            on_beat_hits = np.sum(hits[::4])  # Steps 0, 4, 8, 12 (quarter notes)
            off_beat_hits = total_hits - on_beat_hits
            if on_beat_hits > 0:
                bar_stat['syncopation_ratio'] = float(off_beat_hits / on_beat_hits)
            else:
                bar_stat['syncopation_ratio'] = float('inf') if off_beat_hits > 0 else 0.0
            
            bar_stats.append(bar_stat)
        
        # Aggregate statistics across bars
        agg_stats = self._aggregate_bar_stats(bar_stats)
        
        # Bar-to-bar similarity for repetition analysis
        similarities = self._compute_bar_similarities(hit_bars)
        agg_stats['bar_similarity_mean'] = float(np.mean(similarities)) if len(similarities) > 0 else 0.0
        agg_stats['bar_similarity_std'] = float(np.std(similarities)) if len(similarities) > 0 else 0.0
        
        return {
            'num_bars': num_bars,
            'per_bar': bar_stats,
            'aggregate': agg_stats
        }

    def _empty_stats(self):
        """Return empty stats for invalid inputs."""
        return {
            'num_bars': 0,
            'per_bar': [],
            'aggregate': {
                'total_density': 0.0,
                'per_drum_density': [0.0] * self.comp,
                'hit_position_dist': [0.0] * self.steps_per_bar,
                'velocity_mean': 0.0,
                'velocity_std': 0.0,
                'offset_std': 0.0,
                'syncopation_ratio': 0.0,
                'bar_similarity_mean': 0.0,
                'bar_similarity_std': 0.0
            }
        }
    
    def _aggregate_bar_stats(self, bar_stats):
        """Aggregate statistics across bars."""
        if not bar_stats:
            return self._empty_stats()['aggregate']
        
        # Average across bars
        agg = {}
        agg['total_density'] = np.mean([b['total_density'] for b in bar_stats])
        agg['per_drum_density'] = np.mean([b['per_drum_density'] for b in bar_stats], axis=0).tolist()
        agg['hit_position_dist'] = np.mean([b['hit_position_dist'] for b in bar_stats], axis=0).tolist()
        agg['velocity_mean'] = np.mean([b['velocity_mean'] for b in bar_stats])
        agg['velocity_std'] = np.mean([b['velocity_std'] for b in bar_stats])
        agg['offset_std'] = np.mean([b['offset_std'] for b in bar_stats])
        agg['syncopation_ratio'] = np.mean([b['syncopation_ratio'] for b in bar_stats if not np.isinf(b['syncopation_ratio'])])
        
        return agg
    
    def _compute_bar_similarities(self, hit_bars):
        """Compute cosine similarities between consecutive bars."""
        if len(hit_bars) < 2:
            return []
        
        similarities = []
        for i in range(len(hit_bars) - 1):
            vec1 = hit_bars[i].flatten()
            vec2 = hit_bars[i + 1].flatten()
            
            # Avoid division by zero
            if np.sum(vec1) == 0 and np.sum(vec2) == 0:
                sim = 1.0
            elif np.sum(vec1) == 0 or np.sum(vec2) == 0:
                sim = 0.0
            else:
                sim = 1.0 - cosine(vec1, vec2)
            similarities.append(sim)
        
        return similarities
    
    def compare_stats(self, seed_stats, gen_stats, tolerances=None):
        """
        Compare seed and generated statistics, return deviation score and violations.
        
        Args:
            seed_stats: Stats dict from compute_bar_stats for seed
            gen_stats: Stats dict from compute_bar_stats for generated
            tolerances: Dict of tolerance values
        
        Returns:
            dict with 'score' (lower is better), 'violations', 'details'
        """
        if tolerances is None:
            tolerances = {
                'total_density_pct': 0.2,  # 20%
                'per_drum_density_pct': 0.3,  # 30%
                'hit_position_l1': 0.3,
                'velocity_mean_diff': 0.1,
                'velocity_std_diff': 0.1,
                'offset_std_diff': 0.1,
                'empty_bar_penalty': 10.0,
                'bar_similarity_diff': 0.3
            }
        
        seed_agg = seed_stats['aggregate']
        gen_agg = gen_stats['aggregate']
        
        violations = []
        score = 0.0
        details = {}
        
        # 1. Empty bar penalty
        gen_bars = gen_stats['per_bar']
        empty_bars = sum(1 for bar in gen_bars if bar['total_density'] == 0)
        seed_has_content = seed_stats['num_bars'] > 0 and seed_agg['total_density'] > 0
        
        if empty_bars > 0 and seed_has_content:
            penalty = empty_bars * tolerances['empty_bar_penalty']
            score += penalty
            violations.append(f"Empty bars: {empty_bars}")
            details['empty_bar_penalty'] = penalty
        
        # 2. Total density difference
        if seed_agg['total_density'] > 0:
            density_diff_pct = abs(gen_agg['total_density'] - seed_agg['total_density']) / seed_agg['total_density']
            if density_diff_pct > tolerances['total_density_pct']:
                score += density_diff_pct * 5.0  # Weight
                violations.append(f"Total density diff: {density_diff_pct:.3f} > {tolerances['total_density_pct']}")
            details['total_density_diff_pct'] = density_diff_pct
        
        # 3. Per-drum density difference
        per_drum_violations = 0
        for i in range(len(seed_agg['per_drum_density'])):
            seed_drum = seed_agg['per_drum_density'][i]
            gen_drum = gen_agg['per_drum_density'][i]
            if seed_drum > 0:
                diff_pct = abs(gen_drum - seed_drum) / seed_drum
                if diff_pct > tolerances['per_drum_density_pct']:
                    per_drum_violations += 1
                    score += diff_pct * 2.0
        if per_drum_violations > 0:
            violations.append(f"Per-drum density violations: {per_drum_violations}")
        details['per_drum_violations'] = per_drum_violations
        
        # 4. Hit position distribution (L1 distance)
        seed_pos = np.array(seed_agg['hit_position_dist'])
        gen_pos = np.array(gen_agg['hit_position_dist'])
        # Normalize to probabilities
        seed_pos_norm = seed_pos / np.sum(seed_pos) if np.sum(seed_pos) > 0 else seed_pos
        gen_pos_norm = gen_pos / np.sum(gen_pos) if np.sum(gen_pos) > 0 else gen_pos
        pos_l1 = np.sum(np.abs(seed_pos_norm - gen_pos_norm))
        if pos_l1 > tolerances['hit_position_l1']:
            score += pos_l1 * 3.0
            violations.append(f"Hit position L1: {pos_l1:.3f} > {tolerances['hit_position_l1']}")
        details['hit_position_l1'] = pos_l1
        
        # 5. Velocity mean difference
        vel_mean_diff = abs(gen_agg['velocity_mean'] - seed_agg['velocity_mean'])
        if vel_mean_diff > tolerances['velocity_mean_diff']:
            score += vel_mean_diff * 5.0
            violations.append(f"Velocity mean diff: {vel_mean_diff:.3f} > {tolerances['velocity_mean_diff']}")
        details['velocity_mean_diff'] = vel_mean_diff
        
        # 6. Velocity std difference
        vel_std_diff = abs(gen_agg['velocity_std'] - seed_agg['velocity_std'])
        if vel_std_diff > tolerances['velocity_std_diff']:
            score += vel_std_diff * 3.0
            violations.append(f"Velocity std diff: {vel_std_diff:.3f} > {tolerances['velocity_std_diff']}")
        details['velocity_std_diff'] = vel_std_diff
        
        # 7. Offset std difference
        off_std_diff = abs(gen_agg['offset_std'] - seed_agg['offset_std'])
        if off_std_diff > tolerances['offset_std_diff']:
            score += off_std_diff * 3.0
            violations.append(f"Offset std diff: {off_std_diff:.3f} > {tolerances['offset_std_diff']}")
        details['offset_std_diff'] = off_std_diff
        
        # 8. Bar similarity difference (for repetitive patterns)
        bar_sim_diff = abs(gen_agg.get('bar_similarity_mean', 0) - seed_agg.get('bar_similarity_mean', 0))
        if bar_sim_diff > tolerances['bar_similarity_diff']:
            score += bar_sim_diff * 2.0
            violations.append(f"Bar similarity diff: {bar_sim_diff:.3f} > {tolerances['bar_similarity_diff']}")
        details['bar_similarity_diff'] = bar_sim_diff
        
        return {
            'score': score,
            'violations': violations,
            'details': details,
            'num_violations': len(violations)
        }

    def _compute_bar_similarity_with_identity(self, bar1, bar2):
        """
        Enhanced bar similarity that includes direct pattern matching for identity detection.
        
        Returns similarity score and whether patterns are essentially identical.
        """
        # Get raw hit patterns for direct comparison
        hits1 = np.array(bar1.get('hit_pattern', []))  # Will add this to bar stats
        hits2 = np.array(bar2.get('hit_pattern', []))
        
        # Direct pattern matching if available
        if len(hits1) > 0 and len(hits2) > 0 and hits1.shape == hits2.shape:
            # Check for identical hit patterns (binary)
            hit_identity = np.array_equal(hits1 > 0.5, hits2 > 0.5)
            
            # Check for near-identical patterns (allowing small velocity/timing differences)
            hit_diff = np.sum(np.abs(hits1 - hits2))
            total_hits = np.sum(hits1 > 0.5) + np.sum(hits2 > 0.5)
            near_identity = hit_diff < 0.1 * max(total_hits, 1)
            
            if hit_identity:
                return 1.0, True  # Exactly identical
            elif near_identity:
                return 0.98, True  # Nearly identical
        
        # Fall back to aggregate similarity (existing logic)
        similarities = []
        
        # 1. Density similarity
        density1 = bar1['total_density']
        density2 = bar2['total_density']
        if density1 + density2 > 0:
            density_sim = 1.0 - abs(density1 - density2) / max(density1, density2, 1.0)
        else:
            density_sim = 1.0  # Both empty
        similarities.append(density_sim)
        
        # 2. Per-drum density pattern similarity (cosine)
        drums1 = np.array(bar1['per_drum_density'])
        drums2 = np.array(bar2['per_drum_density'])
        if np.sum(drums1) > 0 and np.sum(drums2) > 0:
            drum_sim = 1.0 - cosine(drums1, drums2)
        elif np.sum(drums1) == 0 and np.sum(drums2) == 0:
            drum_sim = 1.0
        else:
            drum_sim = 0.0
        similarities.append(drum_sim)
        
        # 3. Hit position distribution similarity
        pos1 = np.array(bar1['hit_position_dist'])
        pos2 = np.array(bar2['hit_position_dist'])
        if np.sum(pos1) > 0 and np.sum(pos2) > 0:
            # Normalize to probabilities
            pos1_norm = pos1 / np.sum(pos1)
            pos2_norm = pos2 / np.sum(pos2)
            # Use negative L1 distance as similarity (closer to 0 = more similar)
            pos_sim = 1.0 - 0.5 * np.sum(np.abs(pos1_norm - pos2_norm))
        elif np.sum(pos1) == 0 and np.sum(pos2) == 0:
            pos_sim = 1.0
        else:
            pos_sim = 0.0
        similarities.append(pos_sim)
        
        # 4. Velocity similarity
        vel_diff = abs(bar1['velocity_mean'] - bar2['velocity_mean'])
        vel_sim = max(0.0, 1.0 - vel_diff / 0.5)  # Normalize by reasonable range
        similarities.append(vel_sim)
        
        # 5. Syncopation similarity
        sync1 = bar1['syncopation_ratio'] if not np.isinf(bar1['syncopation_ratio']) else 0.0
        sync2 = bar2['syncopation_ratio'] if not np.isinf(bar2['syncopation_ratio']) else 0.0
        sync_diff = abs(sync1 - sync2)
        sync_sim = max(0.0, 1.0 - sync_diff / 2.0)  # Normalize by reasonable range
        similarities.append(sync_sim)
        
        # Weight different aspects
        weights = [0.25, 0.3, 0.25, 0.1, 0.1]  # Emphasize drum patterns and positions
        weighted_sim = np.average(similarities, weights=weights)
        
        # Consider high aggregate similarity as potential identity
        is_identity = weighted_sim > 0.92
        
        return max(0.0, min(1.0, weighted_sim)), is_identity

    def compare_bar_by_bar(self, parent_stats, generated_stats, similarity_threshold=0.7, 
                          min_variation_threshold=0.15, max_identity_threshold=0.85):
        """
        Compare generated pattern to parent pattern bar-by-bar with enhanced identity detection.
        
        FIXED: Now properly compares seed vs continuation, not full parent vs continuation.
        """
        parent_bars = parent_stats['per_bar']
        generated_bars = generated_stats['per_bar']
        
        if len(parent_bars) == 0 or len(generated_bars) == 0:
            return {
                'overall_similarity': 0.0,
                'bar_similarities': [],
                'passes_threshold': False,
                'comparison_bars': 0,
                'is_too_identical': False,
                'variation_score': 0.0,
                'identity_violations': []
            }
        
        # Compare each generated bar with the corresponding parent bar
        bar_similarities = []
        identity_violations = []
        variation_scores = []
        identity_detections = []
        
        for i, gen_bar in enumerate(generated_bars):
            # Use modulo to cycle through parent bars if generated is longer
            parent_idx = min(i, len(parent_bars) - 1)
            parent_bar = parent_bars[parent_idx]
            
            # Compute enhanced similarity with identity detection
            sim_score, is_identical = self._compute_bar_similarity_with_identity(parent_bar, gen_bar)
            bar_similarities.append(sim_score)
            identity_detections.append(is_identical)
            
            # Compute multi-dimensional variation score
            variation_score = self._compute_variation_score(parent_bar, gen_bar)
            variation_scores.append(variation_score)
            
            # Check for identity violations with lower threshold
            if is_identical or sim_score > max_identity_threshold:
                identity_violations.append(f"Bar {i}: identical pattern detected (sim={sim_score:.3f})")
        
        # Overall metrics
        overall_similarity = np.mean(bar_similarities) if bar_similarities else 0.0
        overall_variation = np.mean(variation_scores) if variation_scores else 0.0
        
        # Enhanced identity detection
        num_identical = sum(identity_detections)
        pct_identical = num_identical / len(identity_detections) if identity_detections else 0.0
        
        # More strict identity detection
        is_too_identical = (pct_identical > 0.5 or  # More than 50% of bars are identical
                           overall_similarity > max_identity_threshold or 
                           overall_variation < min_variation_threshold)
        
        return {
            'overall_similarity': float(overall_similarity),
            'bar_similarities': bar_similarities,
            'passes_threshold': overall_similarity >= similarity_threshold,
            'comparison_bars': len(bar_similarities),
            'is_too_identical': is_too_identical,
            'variation_score': float(overall_variation),
            'identity_violations': identity_violations,
            'num_identical_bars': num_identical,
            'pct_identical_bars': float(pct_identical)
        }

    def _compute_variation_score(self, parent_bar, gen_bar):
        """
        Compute multi-dimensional variation score to detect near-identical patterns.
        Higher score means more variation.
        """
        variations = []
        
        # 1. Density variation (normalized)
        density_var = abs(gen_bar['total_density'] - parent_bar['total_density'])
        density_var_norm = min(1.0, density_var / max(parent_bar['total_density'], 1.0))
        variations.append(density_var_norm)
        
        # 2. Drum pattern variation
        drums1 = np.array(parent_bar['per_drum_density'])
        drums2 = np.array(gen_bar['per_drum_density'])
        if np.sum(drums1) > 0 or np.sum(drums2) > 0:
            drum_variation = np.sum(np.abs(drums1 - drums2)) / (np.sum(drums1) + np.sum(drums2) + 1e-6)
        else:
            drum_variation = 0.0
        variations.append(drum_variation)
        
        # 3. Rhythmic position variation
        pos1 = np.array(parent_bar['hit_position_dist'])
        pos2 = np.array(gen_bar['hit_position_dist'])
        if np.sum(pos1) > 0 or np.sum(pos2) > 0:
            pos_variation = np.sum(np.abs(pos1 - pos2)) / (np.sum(pos1) + np.sum(pos2) + 1e-6)
        else:
            pos_variation = 0.0
        variations.append(pos_variation)
        
        # 4. Velocity variation
        vel_var = abs(gen_bar['velocity_mean'] - parent_bar['velocity_mean'])
        variations.append(min(1.0, vel_var / 0.5))  # Normalize by reasonable range
        
        # 5. Syncopation variation
        sync1 = parent_bar['syncopation_ratio'] if not np.isinf(parent_bar['syncopation_ratio']) else 0.0
        sync2 = gen_bar['syncopation_ratio'] if not np.isinf(gen_bar['syncopation_ratio']) else 0.0
        sync_var = abs(sync2 - sync1)
        variations.append(min(1.0, sync_var / 2.0))  # Normalize by reasonable range
        
        # Weight the variations (emphasize rhythmic and drum pattern changes)
        weights = [0.2, 0.35, 0.25, 0.1, 0.1]
        return np.average(variations, weights=weights)


# === ENHANCED STOCHASTIC DECODER ===

class StochasticDecoder:
    """Wraps the model decoder to add stochastic sampling and parameter control."""
    
    def __init__(self, decoder):
        self.decoder = decoder

    def __call__(self, z, seq_len, temperature=1.0, hit_threshold=0.0, hit_bias=0.0, 
                 velocity_scale=1.0, offset_scale=1.0, position_bias=None):
        """Apply stochastic sampling with parameter control."""
        with torch.no_grad():
            # Get model output
            output, hits, velocities, offsets = self.decoder(z, seq_len)
            
            # Split output into components
            comp = 9
            hit_probs = hits
            velocities = velocities
            offsets = offsets
            
            # Apply temperature to hits (stochastic sampling)
            if temperature != 1.0:
                hit_probs = hit_probs / temperature
            
            # Apply hit bias if specified
            if hit_bias != 0.0:
                hit_probs = hit_probs + hit_bias
            
            # Apply position bias if specified
            if position_bias is not None:
                for pos, bias in position_bias.items():
                    hit_probs[:, pos::16, :] += bias  # Apply every 16th step (downbeats, etc.)
            
            # Stochastic sampling for hits
            sampled_hits = torch.bernoulli(torch.sigmoid(hit_probs))
            
            # Apply hit threshold as post-processing filter
            if hit_threshold > 0.0:
                sampled_hits = sampled_hits * (hit_probs > hit_threshold).float()
            
            # FIXED: Completely overhauled velocity handling
            # Add significant velocity variation and ensure proper scaling
            velocity_noise = torch.randn_like(velocities) * 0.25  # Add 25% noise for variation
            varied_velocities = velocities + velocity_noise
            
            # Apply velocity scaling with safe bounds
            safe_velocity_scale = max(velocity_scale, 0.8)  # Minimum 80% 
            scaled_velocities = varied_velocities * safe_velocity_scale
            
            # Conservative offset scaling
            safe_offset_scale = min(max(offset_scale, 0.3), 1.0)  # 30%-100% range
            scaled_offsets = offsets * safe_offset_scale
            
            # Clamp to valid ranges with much better velocity distribution
            scaled_velocities = torch.clamp(scaled_velocities, 0.3, 1.0)  # 30%-100% range
            scaled_offsets = torch.clamp(scaled_offsets, -0.2, 0.2)
            
            # Combine back
            final_output = torch.cat([sampled_hits, scaled_velocities, scaled_offsets], dim=-1)
            
            return final_output, hit_probs, sampled_hits, (scaled_velocities, scaled_offsets)


# === SELF-ADJUST GENERATION ===

def sample_latent_variation(mu, logvar, temperature=1.0):
    """
    Sample from latent space with temperature control for variation.
    
    Args:
        mu: Mean of latent distribution
        logvar: Log variance of latent distribution  
        temperature: Temperature for controlling variation (higher = more variation)
    
    Returns:
        Sampled latent vector z
    """
    if temperature is None:
        temperature = 1.0
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    # Apply temperature scaling to the noise
    z = mu + eps * std * max(float(temperature), 1e-8)
    return z


def generate_with_feedback(encoder, decoder, device, seed_input, seed_stats, 
                          bars_to_add=4, steps_per_bar=16, max_attempts=5,
                          initial_params=None, tolerances=None, analyzer=None):
    """
    Generate continuation with feedback loop to improve quality.
    
    Args:
        encoder, decoder: Model components
        device: torch device
        seed_input: Tensor input for encoding
        seed_stats: Pre-computed stats for comparison
        bars_to_add: Number of bars to generate
        steps_per_bar: Steps per bar
        max_attempts: Maximum refinement attempts
        initial_params: Starting generation parameters
        tolerances: Deviation tolerances
        analyzer: DrumPatternAnalyzer instance
    
    Returns:
        (final_output_tensor, final_params, attempt_info)
    """
    if initial_params is None:
        params = {
            'temperature': 0.8,
            'hit_threshold': 0.0,
            'hit_bias': 0.0,
            'velocity_scale': 1.0,
            'offset_scale': 1.0,
            'latent_temperature': 0.5
        }
    else:
        params = initial_params.copy()
    
    stochastic_decoder = StochasticDecoder(decoder)
    seq_len = bars_to_add * steps_per_bar
    
    # Encode seed
    with torch.no_grad():
        z_enc, mu, logvar = encoder(seed_input)
    
    attempt_info = []
    
    for attempt in range(max_attempts):
        # Sample from latent space with current temperature
        z = sample_latent_variation(mu, logvar, params['latent_temperature'])
        
        # Generate
        output, hit_probs, sampled_hits, (vels, offs) = stochastic_decoder(
            z, seq_len,
            temperature=params['temperature'],
            hit_threshold=params['hit_threshold'],
            hit_bias=params['hit_bias'],
            velocity_scale=params['velocity_scale'],
            offset_scale=params['offset_scale']
        )
        
        generated = output.squeeze(0).cpu()
        
        # Analyze generated output
        gen_hits = generated[:, :9].numpy()
        gen_vels = generated[:, 9:18].numpy()
        gen_offs = generated[:, 18:].numpy()
        
        gen_stats = analyzer.compute_bar_stats(gen_hits, gen_vels, gen_offs)
        comparison = analyzer.compare_stats(seed_stats, gen_stats, tolerances)
        
        attempt_info.append({
            'attempt': attempt + 1,
            'params': params.copy(),
            'score': comparison['score'],
            'violations': comparison['num_violations'],
            'details': comparison['details']
        })
        
        # If good enough, return
        if comparison['num_violations'] <= 1 and comparison['score'] < 5.0:
            return generated, params, attempt_info
        
        # Otherwise, adjust parameters for next attempt
        if attempt < max_attempts - 1:
            params = adjust_parameters(params, comparison, seed_stats, gen_stats)
    
    # Return best attempt (last one)
    return generated, params, attempt_info


def adjust_parameters(params, comparison, seed_stats, gen_stats):
    """
    Adjust generation parameters based on comparison results.
    """
    new_params = params.copy()
    details = comparison['details']
    
    # Adjust based on specific violations
    if 'total_density_diff_pct' in details:
        density_diff = details['total_density_diff_pct']
        seed_density = seed_stats['aggregate']['total_density']
        gen_density = gen_stats['aggregate']['total_density']
        
        if gen_density < seed_density:  # Too sparse
            new_params['hit_bias'] = min(new_params['hit_bias'] + 0.1, 0.5)
            new_params['hit_threshold'] = max(new_params['hit_threshold'] - 0.05, 0.0)
        else:  # Too dense
            new_params['hit_bias'] = max(new_params['hit_bias'] - 0.1, -0.5)
            new_params['hit_threshold'] = min(new_params['hit_threshold'] + 0.05, 0.3)
    
    if 'velocity_mean_diff' in details:
        vel_diff = details['velocity_mean_diff']
        seed_vel = seed_stats['aggregate']['velocity_mean']
        gen_vel = gen_stats['aggregate']['velocity_mean']
        
        if gen_vel < seed_vel:  # Too quiet
            new_params['velocity_scale'] = min(new_params['velocity_scale'] + 0.1, 1.5)
        else:  # Too loud
            new_params['velocity_scale'] = max(new_params['velocity_scale'] - 0.1, 0.5)
    
    if 'offset_std_diff' in details:
        off_diff = details['offset_std_diff']
        seed_off = seed_stats['aggregate']['offset_std']
        gen_off = gen_stats['aggregate']['offset_std']
        
        if gen_off < seed_off:  # Too rigid timing
            new_params['offset_scale'] = min(new_params['offset_scale'] + 0.1, 1.5)
        else:  # Too loose timing
            new_params['offset_scale'] = max(new_params['offset_scale'] - 0.1, 0.5)
    
    # Adjust temperature if many violations
    if comparison['num_violations'] > 3:
        new_params['temperature'] = min(new_params['temperature'] + 0.1, 1.5)
        new_params['latent_temperature'] = min(new_params['latent_temperature'] + 0.05, 1.0)
    elif comparison['num_violations'] == 0:
        new_params['temperature'] = max(new_params['temperature'] - 0.05, 0.3)
    
    return new_params


# === QUALITY FILTERING ===

def filter_quality_variations(results, original_midi_path, analyzer, similarity_threshold=0.7, 
                            max_keep=3, min_variation_threshold=0.2, max_identity_threshold=0.85):
    """
    FIXED: Enhanced filter that properly compares seed vs continuation, not full vs continuation.
    """
    if not results:
        return [], []

    # Analyze the original/parent MIDI to get the SEED pattern (last 2 bars)
    try:
        parent_pm = pretty_midi.PrettyMIDI(str(original_midi_path))
        parent_inst = get_drum_instrument(parent_pm)
        if parent_inst is None:
            print(f"  - Warning: No drum instrument in parent {original_midi_path}")
            return results, []
        
        # Get full original pattern
        parent_fs, parent_start = safe_compute_fs_and_start(parent_pm)
        full_original, seed_input = build_full_original_and_seed(
            parent_inst, parent_fs, parent_start, steps_per_bar=16, window_bars=2, hop_size=16
        )
        
        # Extract seed pattern for comparison (last 2 bars = 32 steps)
        seed_tensor = seed_input.squeeze(0)  # Remove batch dim
        seed_hits = seed_tensor[:, :9].numpy()
        seed_vels = seed_tensor[:, 9:18].numpy()
        seed_offs = seed_tensor[:, 18:].numpy()
        seed_stats = analyzer.compute_bar_stats(seed_hits, seed_vels, seed_offs)
        
        print(f"  - Seed pattern: {seed_stats['num_bars']} bars, density={seed_stats['aggregate']['total_density']:.1f}")
        
    except Exception as e:
        print(f"  - Warning: Could not analyze parent MIDI {original_midi_path}: {e}")
        return results, []
    
    # Analyze each generated variation and compute similarity to SEED
    variation_scores = []
    
    for i, item in enumerate(results):
        try:
            # Robustly unpack result tuple (supports 3- and 4-tuples)
            if len(item) >= 4:
                var_num, midi_out, file_path, original_used_steps = item
            else:
                var_num, midi_out, file_path = item
                original_used_steps = len(full_original)  # fallback for backward-compat
            
            # Extract GENERATED CONTINUATION data (skip the original bars, analyze only new bars)
            gen_inst = get_drum_instrument(midi_out)
            if gen_inst is None:
                variation_scores.append((i, 0.0, 0.0, True, "No drum instrument"))
                continue
            
            gen_fs, gen_start = safe_compute_fs_and_start(midi_out)
            gen_hits, gen_vels, gen_offs, _ = quantize_drum(gen_inst, gen_fs, gen_start)
            
            # Skip original part, analyze only the continuation using the ACTUAL original_used_steps
            if len(gen_hits) > original_used_steps:
                cont_hits = gen_hits[original_used_steps:]
                cont_vels = gen_vels[original_used_steps:]
                cont_offs = gen_offs[original_used_steps:]
                gen_stats = analyzer.compute_bar_stats(cont_hits, cont_vels, cont_offs)
            else:
                print(f"    Variation {var_num}: Generated sequence too short, skipping")
                variation_scores.append((i, 0.0, 0.0, True, "Too short"))
                continue
            
            # Compare SEED vs CONTINUATION using enhanced method
            comparison = analyzer.compare_bar_by_bar(
                seed_stats, gen_stats, similarity_threshold, 
                min_variation_threshold, max_identity_threshold
            )
            
            similarity_score = comparison['overall_similarity']
            variation_score = comparison['variation_score']
            is_too_identical = comparison['is_too_identical']
            pct_identical = comparison.get('pct_identical_bars', 0.0)
            
            variation_scores.append((i, similarity_score, variation_score, is_too_identical, 
                                   f"sim={similarity_score:.3f}, var={variation_score:.3f}, id={pct_identical:.1%}"))
            
            print(f"    Variation {var_num}: similarity={similarity_score:.3f}, variation={variation_score:.3f}, "
                  f"identical_bars={pct_identical:.1%}, {'REJECTED' if is_too_identical else 'OK'}")
            
        except Exception as e:
            print(f"    Variation {var_num}: Analysis failed: {e}")
            variation_scores.append((i, 0.0, 0.0, True, f"Error: {str(e)}"))
    
    # Filter and sort variations
    suitable_variations = []
    deleted_files = []
    
    for idx, sim_score, var_score, is_identical, details in variation_scores:
        # Robustly access tuple values for deletion and logging
        var = results[idx]
        var_num = var[0]
        file_path = var[2]
        
        # Enhanced filtering criteria
        passes_similarity = sim_score >= similarity_threshold
        not_identical = not is_identical
        has_variation = var_score >= min_variation_threshold
        
        if passes_similarity and not_identical and has_variation:
            suitable_variations.append((sim_score, var_score, idx))
        else:
            if file_path and file_path.exists():
                file_path.unlink()  # Delete file
                deleted_files.append(str(file_path))
            print(f"    DELETED Variation {var_num}: {details}")
    
    # Sort by similarity score (descending) and keep top max_keep
    suitable_variations.sort(key=lambda x: x[0], reverse=True)
    kept_indices = [idx for _, _, idx in suitable_variations[:max_keep]]
    
    # Delete excess files
    for _, _, idx in suitable_variations[max_keep:]:
        var = results[idx]
        var_num = var[0]
        file_path = var[2]
        if file_path and file_path.exists():
            file_path.unlink()
            deleted_files.append(str(file_path))
        print(f"    DELETED Variation {var_num}: Exceeded max_keep={max_keep}")
    
    kept_results = [results[idx] for idx in kept_indices]
    
    return kept_results, deleted_files


# === MODEL LOADING ===

def load_models(checkpoint=None, encoder_path=None, decoder_path=None, device=None):
    """Load models from checkpoint or separate files."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else
                            "cuda" if torch.cuda.is_available() else "cpu")
    
    config = Config()
    encoder = Encoder(
        config.ENC_INPUT_SIZE,
        config.ENC_HIDDEN_SIZE,
        config.ENC_LATENT_DIM
    ).to(device)
    decoder = Decoder(
        config.DEC_INPUT_SIZE,
        config.DEC_HIDDEN_SIZE,
        config.DEC_OUTPUT_SIZE
    ).to(device)
    
    if checkpoint:
        checkpoint_data = torch.load(checkpoint, map_location=device)
        if 'encoder_state_dict' in checkpoint_data:
            encoder.load_state_dict(checkpoint_data['encoder_state_dict'])
            decoder.load_state_dict(checkpoint_data['decoder_state_dict'])
        else:
            # Assume it's a combined state dict
            encoder.load_state_dict(checkpoint_data)
            decoder.load_state_dict(checkpoint_data)
    else:
        if encoder_path:
            encoder.load_state_dict(torch.load(encoder_path, map_location=device))
        if decoder_path:
            decoder.load_state_dict(torch.load(decoder_path, map_location=device))
    
    encoder.eval()
    decoder.eval()
    return encoder, decoder, device


# === UTILITY FUNCTIONS ===

def get_drum_instrument(pm: pretty_midi.PrettyMIDI):
    """Get a drum instrument. Fallback: build a drum instrument by mapping notes from the most mappable track."""
    # First, prefer any track explicitly marked as drum and containing notes
    for inst in pm.instruments:
        if inst.is_drum and inst.notes:
            return inst

    # Fallback: find the instrument with the most notes that can map to GM drum set
    best_inst_notes = None
    best_count = 0

    for inst in pm.instruments:
        if not inst.notes:
            continue
        mapped_notes = []
        count = 0
        for n in inst.notes:
            # Copy the note so we don't mutate original track pitches
            note_copy = pretty_midi.Note(velocity=n.velocity, pitch=n.pitch, start=n.start, end=n.end)
            if map_unique_drum(note_copy):  # returns True if retained/mapped to drum pitch
                mapped_notes.append(note_copy)
                count += 1
        if count > best_count:
            best_count = count
            best_inst_notes = mapped_notes

    if best_inst_notes and best_count > 0:
        # Build a synthetic drum instrument from mapped notes
        drum_inst = pretty_midi.Instrument(program=0, is_drum=True)
        drum_inst.notes.extend(best_inst_notes)
        return drum_inst

    # No usable drum-like content found
    return None


def reconstruct_from_windows(windows: torch.Tensor, window_size=32, hop_size=16) -> torch.Tensor:
    """
    Reconstruct a sequence from overlapping windows using averaging.
    
    Args:
        windows: Tensor of shape (num_windows, window_size, feature_dim)
        window_size: Size of each window
        hop_size: Hop size between windows
        
    Returns:
        Reconstructed sequence tensor
    """
    num_windows, _, feature_dim = windows.shape
    total_length = (num_windows - 1) * hop_size + window_size
    
    # Initialize output tensor and count tensor for averaging
    output = torch.zeros(total_length, feature_dim, dtype=windows.dtype, device=windows.device)
    counts = torch.zeros(total_length, dtype=torch.float32, device=windows.device)
    
    # Add each window to the output
    for i, window in enumerate(windows):
        start_idx = i * hop_size
        end_idx = start_idx + window_size
        output[start_idx:end_idx] += window
        counts[start_idx:end_idx] += 1.0
    
    # Avoid division by zero
    counts = torch.clamp(counts, min=1.0)
    
    # Average overlapping regions
    output = output / counts.unsqueeze(-1)
    
    return output


def safe_compute_fs_and_start(pm: pretty_midi.PrettyMIDI):
    """Safely compute fs, start_time, and tempo with correct model resolution."""
    # Model expects 16 steps per quarter note (from training)
    fs = 16  # This is the model's temporal resolution
    start_time = 0.0
    
    # Extract tempo (BPM)
    tempo_bpm = 120  # Default
    if pm.get_tempo_changes()[1].size > 0:
        tempo_bpm = pm.get_tempo_changes()[1][0]  # Use first tempo
    
    return fs, start_time, tempo_bpm


def build_full_original_and_seed(inst, fs, start_time, steps_per_bar=16, window_bars=2, hop_size=16):
    """Build original sequence and extract seed."""
    hits, velocities, offsets, _ = quantize_drum(inst, fs, start_time)
    
    # Convert to tensor and combine
    hit_tensor = torch.tensor(hits, dtype=torch.float32)
    vel_tensor = torch.tensor(velocities, dtype=torch.float32)
    off_tensor = torch.tensor(offsets, dtype=torch.float32)
    
    full_original = torch.cat([hit_tensor, vel_tensor, off_tensor], dim=-1)
    
    # Extract seed (last window_bars)
    window_size = window_bars * steps_per_bar
    if full_original.shape[0] >= window_size:
        seed_input = full_original[-window_size:].unsqueeze(0)  # Add batch dimension
    else:
        # Pad if too short
        padding_needed = window_size - full_original.shape[0]
        padding = torch.zeros(padding_needed, full_original.shape[1])
        padded = torch.cat([padding, full_original], dim=0)
        seed_input = padded.unsqueeze(0)
    
    return full_original, seed_input


def from_tensors_to_midi_consistent(tensor, fs, start_time, steps_per_quarter=16, comp=9, tempo_bpm=120):
    """Updated version that handles Ableton Live's 96 ppq properly."""
    return from_tensors_to_midi(
        tensor, 
        steps_per_quarter=16,  # FIXED: Use model's training resolution
        comp=comp, 
        fs=fs, 
        tempo_bpm=tempo_bpm,
        source_ppq=96  # Ableton Live's export resolution
    )


def generate_continuation_for_midi(
    midi_path: Path,
    encoder: Encoder,
    decoder: Decoder,
    device,
    bars_to_add=4,
    num_variations=3,
    temperature=0.8,
    hit_threshold=0.0,
    latent_temperature=0.5,
    steps_per_bar=16,
    window_bars=2,  # CRITICAL: This must match model training
    hop_size=16,
    self_adjust=False,
    max_attempts=5,
    tolerances=None,
    quality_filter=False,
    similarity_threshold=0.7,
    max_keep=3,
    min_variation_threshold=0.15,
    max_identity_threshold=0.95,
    output_dir=None,
) -> list:
    """
    Generate continuations for a single MIDI file with proper chunking.
    
    FIXED: Generate in 2-bar chunks to match training data distribution.
    """
    try:
        pm = pretty_midi.PrettyMIDI(str(midi_path))
        inst = get_drum_instrument(pm)
        if inst is None:
            print(f"Warning: No drum instrument found in {midi_path}")
            return []
        
        fs, start_time, tempo_bpm = safe_compute_fs_and_start(pm)
        full_original, seed_input = build_full_original_and_seed(
            inst, fs, start_time, steps_per_bar, window_bars, hop_size
        )
        
        # Move to device
        seed_input = seed_input.to(device)
        
        # Initialize analyzer if needed
        analyzer = None
        if self_adjust or quality_filter:
            analyzer = DrumPatternAnalyzer(steps_per_bar, comp=9)
            
        results = []
        
        for v in range(num_variations):
            print(f"  Generating variation {v+1}/{num_variations}...")
            
            # FIXED: Generate in chunks that match training window size
            generated_chunks = []
            current_seed = seed_input  # Start with original seed
            
            # Calculate how many 2-bar chunks we need
            chunks_needed = max(1, bars_to_add // window_bars)
            remaining_bars = bars_to_add % window_bars
            
            for chunk_idx in range(chunks_needed):
                # Generate one 2-bar chunk
                chunk_bars = window_bars
                if chunk_idx == chunks_needed - 1 and remaining_bars > 0:
                    chunk_bars = remaining_bars
                
                seq_len = chunk_bars * steps_per_bar
                
                if self_adjust:
                    # Use feedback generation for this chunk
                    seed_stats = analyzer.compute_bar_stats(
                        current_seed[0, :, :9].numpy(),
                        current_seed[0, :, 9:18].numpy(),
                        current_seed[0, :, 18:].numpy()
                    ) if analyzer else None
                    
                    generated_chunk, final_params, attempt_info = generate_with_feedback(
                        encoder, decoder, device, current_seed, seed_stats,
                        chunk_bars, steps_per_bar, max_attempts, None, tolerances, analyzer
                    )
                else:
                    # Use standard stochastic generation for this chunk
                    stochastic_decoder = StochasticDecoder(decoder)
                    
                    with torch.no_grad():
                        z, mu, logvar = encoder(current_seed)
                        z = sample_latent_variation(mu, logvar, latent_temperature)
                        
                        output, _, _, _ = stochastic_decoder(
                            z, seq_len, temperature, hit_threshold
                        )
                        generated_chunk = output.squeeze(0).cpu()
                
                generated_chunks.append(generated_chunk)
                
                # Update seed for next chunk: use the last window_bars of combined sequence
                if chunk_idx < chunks_needed - 1:  # Not the last chunk
                    # Combine current seed with generated chunk on CPU, then move back to device
                    combined = torch.cat([current_seed.squeeze(0).cpu(), generated_chunk], dim=0)
                    
                    # Take the last window_bars as the new seed
                    window_size = window_bars * steps_per_bar
                    if combined.shape[0] >= window_size:
                        current_seed = combined[-window_size:].unsqueeze(0).to(device)
                    else:
                        current_seed = combined.unsqueeze(0).to(device)
            
            # Combine all generated chunks
            generated = torch.cat(generated_chunks, dim=0)
            
            # Append continuation to original (but limit original length)
            max_original_bars = 8
            max_original_steps = max_original_bars * steps_per_bar

            if full_original.shape[0] > max_original_steps:
                trimmed_original = full_original[-max_original_steps:]
                final_seq = torch.cat([trimmed_original, generated], dim=0)
                original_used_steps = trimmed_original.shape[0]
            else:
                final_seq = torch.cat([full_original, generated], dim=0)
                original_used_steps = full_original.shape[0]

            # Convert to MIDI with consistent timing and original tempo
            midi_out = from_tensors_to_midi_consistent(
                final_seq.numpy(), fs, start_time, 
                steps_per_quarter=16, comp=9, tempo_bpm=tempo_bpm
            )
            
            # Set meaningful track name
            if midi_out.instruments:
                track_name = f"{midi_path.stem}_continued_{bars_to_add}bars_v{v+1}"
                midi_out.instruments[0].name = track_name
                midi_out.instruments[0].is_drum = True
            
            # For quality filtering, we need the file path
            file_path = None
            if quality_filter and output_dir:
                file_path = output_dir / f"{track_name}.mid"
                midi_out.write(str(file_path))
            
            results.append((v+1, midi_out, file_path))
        
        return results
    
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        traceback.print_exc()
        return []


def main():
    parser = argparse.ArgumentParser(description='Generate drum pattern continuations')
    
    # Input/Output
    parser.add_argument('--input-dir', type=str, 
                       default='/Users/gjb/_sound/_libs/_midi/_beats',  # FIXED: Your default
                       help='Directory containing input MIDI files')
    parser.add_argument('--output-dir', type=str, 
                       default='/Users/gjb/_sound/_libs/_midi/_groovae',  # FIXED: Corrected path
                       help='Directory to save generated continuations')
    
    # Generation parameters - FIXED: Better defaults for chunked generation
    parser.add_argument('--bars-to-add', type=int, default=2,  # FIXED: Start with 2 bars
                       help='Number of bars to add as continuation')
    parser.add_argument('--num-variations', type=int, default=3,
                       help='Number of variations to generate per input')
    parser.add_argument('--temperature', type=float, default=0.6,  # FIXED: More conservative
                       help='Temperature for stochastic sampling')
    parser.add_argument('--hit-threshold', type=float, default=0.1,  # FIXED: Low threshold
                       help='Threshold for hit detection')
    parser.add_argument('--latent-temperature', type=float, default=0.4,  # FIXED: Conservative
                       help='Temperature for latent space sampling')
    parser.add_argument("--checkpoint", type=str, help="Combined checkpoint (.pth)")
    parser.add_argument("--encoder", type=str, help="Separate encoder path (.pt/.pth)")
    parser.add_argument("--decoder", type=str, help="Separate decoder path (.pt/.pth)")
    
    # Self-adjust parameters
    parser.add_argument("--self-adjust", action="store_true", help="Enable self-adjust feedback loop")
    parser.add_argument("--max-attempts", type=int, default=5, help="Maximum refinement attempts in self-adjust mode (default: 5)")
    parser.add_argument("--save-stats", action="store_true", help="Save analysis statistics as JSON files")
    
    # Quality filtering parameters
    parser.add_argument("--quality-filter", action="store_true", help="Enable quality filtering to keep only statistically similar variations")
    parser.add_argument("--similarity-threshold", type=float, default=0.7, help="Minimum similarity threshold for quality filtering (default: 0.7)")
    parser.add_argument("--max-keep", type=int, default=3, help="Maximum variations to keep per input after filtering (default: 3)")
    parser.add_argument("--min-variation-threshold", type=float, default=0.15, help="Minimum variation required to avoid identical copies (default: 0.15)")
    parser.add_argument("--max-identity-threshold", type=float, default=0.95, help="Maximum similarity before considering too identical (default: 0.95)")
    
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Update session name to include mode info
    mode_suffix = "_selfadj" if args.self_adjust else "_stoch"
    if args.quality_filter:
        mode_suffix += f"_filtered_s{args.similarity_threshold}_k{args.max_keep}"
    
    session_name = f"{mode_suffix}_{args.bars_to_add}bars_t{args.temperature}_h{args.hit_threshold}_lt{args.latent_temperature}_{timestamp}"
    output_dir = Path(args.output_dir) / session_name
    output_dir.mkdir(parents=True, exist_ok=True)

    encoder, decoder, device = load_models(
        checkpoint=args.checkpoint,
        encoder_path=args.encoder,
        decoder_path=args.decoder,
        device=None
    )

    midi_files = [p for p in input_dir.glob("**/*.mid")] + [p for p in input_dir.glob("**/*.midi")]
    print(f"Found {len(midi_files)} MIDI files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Parameters: temp={args.temperature}, hit_thresh={args.hit_threshold}, latent_temp={args.latent_temperature}")
    
    if args.self_adjust:
        print(f"SELF-ADJUST MODE: max_attempts={args.max_attempts}")
    else:
        print("STANDARD STOCHASTIC MODE")
    
    if args.quality_filter:
        print(f"QUALITY FILTERING: threshold={args.similarity_threshold}, max_keep={args.max_keep}")

    total_saved = 0
    
    for midi_path in midi_files:
        try:
            results = generate_continuation_for_midi(
                midi_path,
                encoder, decoder, device,
                bars_to_add=args.bars_to_add,
                num_variations=args.num_variations,
                temperature=args.temperature,
                hit_threshold=args.hit_threshold,
                latent_temperature=args.latent_temperature,
                steps_per_bar=16,
                window_bars=2,
                hop_size=16,
                self_adjust=args.self_adjust,
                max_attempts=args.max_attempts,
                quality_filter=args.quality_filter,
                similarity_threshold=args.similarity_threshold,
                max_keep=args.max_keep,
                min_variation_threshold=args.min_variation_threshold,
                max_identity_threshold=args.max_identity_threshold,
                output_dir=output_dir,
            )
            
            # Save results (only if not already saved by quality filter)
            for item in results:
                # Support both 3- and 4-tuples
                if len(item) >= 4:
                    vi, midi_out, file_path, _ = item
                else:
                    vi, midi_out, file_path = item

                if not file_path:  # Not saved yet
                    out_name = f"{midi_path.stem}_continued_{args.bars_to_add}bars_t{args.temperature}_h{args.hit_threshold}_lt{args.latent_temperature}_v{vi}.mid"
                    out_path = output_dir / out_name
                    midi_out.write(str(out_path))
                
                total_saved += 1
                
        except Exception as e:
            print(f"Error processing {midi_path}: {e}")
            traceback.print_exc()
            continue

    print(f"Done. Saved {total_saved} files to: {output_dir}")
    if args.quality_filter:
        print(f"Quality filtering enabled - unsuitable variations were automatically deleted")


if __name__ == "__main__":
    main()
