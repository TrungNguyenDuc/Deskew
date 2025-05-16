# -*- coding: utf-8 -*-
"""
Created on Thu May  8 22:08:26 2025

@author: S233755
"""

# --- COMMON IMPORTS ---
import numpy as np
import cupyx.scipy.ndimage as ndi_cp
import cupy as cp  # CuPy for GPU acceleration
import tifffile
import os
import time
import math
from scipy.ndimage import rotate, zoom, gaussian_filter
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk, messagebox
from PIL import Image, ImageTk, ImageOps
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, minimize_scalar
from skimage.measure import label, regionprops
from skimage.exposure import rescale_intensity
from scipy.fft import fftn, fftshift, ifftn, ifftshift
import scipy.signal.windows as ss
import gc
import threading
import queue
from pathlib import Path
import traceback
from scipy.ndimage import label as scipy_label
from scipy.ndimage import gaussian_filter as scipy_gaussian_filter





from scipy.ndimage import rotate, zoom, gaussian_filter


from scipy.ndimage import rotate, zoom as scipy_zoom  # Explicitly alias scipy zoom
from scipy.ndimage import rotate as scipy_rotate    # Explicitly alias scipy rotate

# import tkinter as tk # Assuming GUI parts are separate or handled
# from tkinter import filedialog, scrolledtext, ttk, messagebox
from PIL import Image, ImageTk, ImageOps
import matplotlib
matplotlib.use('Agg')




# --- Pillow Resampling Filter (version compatibility) ---
try:
    LANCZOS_RESAMPLE = Image.Resampling.LANCZOS
except AttributeError:
    try: 
        LANCZOS_RESAMPLE = Image.LANCZOS
    except AttributeError: 
        LANCZOS_RESAMPLE = Image.ANTIALIAS
        print("Warning: Using Image.ANTIALIAS for resizing.")

# --- GPU Memory Management ---
def clear_gpu_memory():
    """Clear CuPy memory to prevent fragmentation and release unused blocks."""
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        # Optional: Synchronize to ensure operations are complete, though free_all_blocks should handle this.
        # cp.cuda.Stream.null.synchronize() 
    except Exception as e:
        # This function should not fail catastrophically itself.
        # If log_queue is available globally or passed, can log here.
        print(f"Minor error during clear_gpu_memory: {e}")


def gpu_shear_image(image_chunk_yxz,
                    original_zsize, cz_ref_0based_full, pixel_shift_z,
                    flip_direction, max_y_offset_full, z_chunk_offset, log_queue=None):
    image_gpu, shear_image_gpu = None, None # Initialize for finally block
    
    # Log initial parameters for this call
    if log_queue:
        log_message_gui(f"  gpu_shear_image ENTRY: chunk_offset={z_chunk_offset}, "
                        f"chunk_shape={image_chunk_yxz.shape if image_chunk_yxz is not None else 'None'}, "
                        f"chunk_dtype={image_chunk_yxz.dtype if image_chunk_yxz is not None else 'None'}, "
                        f"orig_Z={original_zsize}, cz_ref={cz_ref_0based_full}, px_shift={pixel_shift_z:.4f}, "
                        f"flip={flip_direction}, max_y_off={max_y_offset_full}", log_queue, "DEBUG")

    try:
        if image_chunk_yxz is None or image_chunk_yxz.size == 0:
             if log_queue: log_message_gui("  GPU Shear: Input chunk is empty or None. Returning calculated empty array.", log_queue, "INFO")
             expected_y_dim = (image_chunk_yxz.shape[0] if image_chunk_yxz is not None and image_chunk_yxz.ndim > 0 else 0) + 2 * max_y_offset_full
             expected_x_dim = image_chunk_yxz.shape[1] if image_chunk_yxz is not None and image_chunk_yxz.ndim > 1 else 0
             expected_z_dim_chunk = image_chunk_yxz.shape[2] if image_chunk_yxz is not None and image_chunk_yxz.ndim > 2 else 0
             # Determine dtype safely
             dtype_to_use = image_chunk_yxz.dtype if image_chunk_yxz is not None else np.float32 # Default dtype
             if log_queue: log_message_gui(f"  GPU Shear: Empty chunk output shape=({expected_y_dim},{expected_x_dim},{expected_z_dim_chunk}), dtype={dtype_to_use}", log_queue, "DEBUG")
             return np.zeros((expected_y_dim, expected_x_dim, expected_z_dim_chunk), dtype=dtype_to_use)

        current_chunk_ysize = image_chunk_yxz.shape[0]
        current_chunk_xsize = image_chunk_yxz.shape[1]
        current_chunk_zsize = image_chunk_yxz.shape[2]

        if log_queue: log_message_gui(f"  GPU Shear: cp.asarray on chunk with shape {image_chunk_yxz.shape}, dtype {image_chunk_yxz.dtype}", log_queue, "DEBUG")
        image_gpu = cp.asarray(image_chunk_yxz) # This could be a source of TypeError if image_chunk_yxz is bad
        
        padded_ysize = current_chunk_ysize + 2 * max_y_offset_full
        
        if log_queue: log_message_gui(f"  GPU Shear: cp.zeros for shear_image_gpu with shape ({padded_ysize}, {current_chunk_xsize}, {current_chunk_zsize}), dtype {image_gpu.dtype}", log_queue, "DEBUG")
        shear_image_gpu = cp.zeros((padded_ysize, current_chunk_xsize, current_chunk_zsize), dtype=image_gpu.dtype) # Or image_chunk_yxz.dtype
        
        for z_local_idx in range(current_chunk_zsize):
            z_global_idx = z_local_idx + z_chunk_offset
            
            y_offset_float = flip_direction * (z_global_idx - cz_ref_0based_full) * pixel_shift_z
            y_offset = int(matlab_round(y_offset_float)) # Using matlab_round for consistency

            y_s_dest = y_offset + max_y_offset_full
            y_e_dest = y_s_dest + current_chunk_ysize # Use current_chunk_ysize

            y_s_dest_c = max(0, y_s_dest)
            y_e_dest_c = min(padded_ysize, y_e_dest)

            y_s_src_c = max(0, -y_s_dest) # If y_s_dest is negative, src starts from that positive offset
            y_e_src_c = current_chunk_ysize - max(0, y_e_dest - padded_ysize) # How much of src is cut from the end

            if log_queue and z_local_idx < 5: # Log for first few Z slices of the chunk
                log_message_gui(f"    z_loc={z_local_idx}, z_glob={z_global_idx}, y_off={y_offset} (from {y_offset_float:.2f})", log_queue, "DEBUG")
                log_message_gui(f"      Dest: y_s_d={y_s_dest}, y_e_d={y_e_dest} -> Clip: y_s_d_c={y_s_dest_c}, y_e_d_c={y_e_dest_c}", log_queue, "DEBUG")
                log_message_gui(f"      Src:  y_s_s_c={y_s_src_c}, y_e_s_c={y_e_src_c}", log_queue, "DEBUG")

            if y_s_dest_c < y_e_dest_c and y_s_src_c < y_e_src_c: # Ensure clipped ranges are valid
                src_h = y_e_src_c - y_s_src_c
                dest_h = y_e_dest_c - y_s_dest_c

                if log_queue and z_local_idx < 5:
                    log_message_gui(f"        src_h={src_h}, dest_h={dest_h}", log_queue, "DEBUG")

                if src_h == dest_h and src_h > 0:
                    # This is the critical assignment
                    try:
                        # Extract slices to check their shapes explicitly before assignment
                        dest_slice_for_assign = shear_image_gpu[y_s_dest_c:y_e_dest_c, :, z_local_idx]
                        src_slice_to_assign = image_gpu[y_s_src_c:y_e_src_c, :, z_local_idx]

                        if log_queue and z_local_idx < 2: # Even more detail for the very first slices
                             log_message_gui(f"          ASSIGN PRE-CHECK: Dest slice shape: {dest_slice_for_assign.shape}, Src slice shape: {src_slice_to_assign.shape}", log_queue, "DEBUG")
                             log_message_gui(f"                          Dest dtype: {dest_slice_for_assign.dtype}, Src dtype: {src_slice_to_assign.dtype}", log_queue, "DEBUG")


                        if dest_slice_for_assign.shape != src_slice_to_assign.shape:
                            # This should ideally not happen if src_h == dest_h, but it's a critical check
                            err_msg = (f"CRITICAL SHAPE MISMATCH at z_loc={z_local_idx}: "
                                       f"LHS slice shape {dest_slice_for_assign.shape} != RHS slice shape {src_slice_to_assign.shape}. "
                                       f"Calculated heights: src_h={src_h}, dest_h={dest_h}. "
                                       f"Indices: dest=({y_s_dest_c}:{y_e_dest_c}), src=({y_s_src_c}:{y_e_src_c})")
                            if log_queue: log_message_gui(err_msg, log_queue, "ERROR")
                            raise ValueError(err_msg) # Raise a more specific error

                        shear_image_gpu[y_s_dest_c:y_e_dest_c, :, z_local_idx] = src_slice_to_assign
                    
                    except TypeError as te_assign: # Catch TypeError specifically here
                        if log_queue:
                            log_message_gui(f"    ASSIGNMENT TypeError at z_loc={z_local_idx}: {te_assign}", log_queue, "CRITICAL")
                            log_message_gui(f"      LHS slice expr: shear_image_gpu[{y_s_dest_c}:{y_e_dest_c}, :, {z_local_idx}]", log_queue)
                            log_message_gui(f"      RHS slice expr: image_gpu[{y_s_src_c}:{y_e_src_c}, :, {z_local_idx}]", log_queue)
                            if 'dest_slice_for_assign' in locals():
                                log_message_gui(f"      Actual LHS slice shape: {dest_slice_for_assign.shape}, dtype: {dest_slice_for_assign.dtype}", log_queue)
                            if 'src_slice_to_assign' in locals():
                                log_message_gui(f"      Actual RHS slice shape: {src_slice_to_assign.shape}, dtype: {src_slice_to_assign.dtype}", log_queue)
                        raise # Re-raise the TypeError to be caught by the outer handler
                    except Exception as e_assign: # Catch other errors during assignment
                        if log_queue: log_message_gui(f"    ASSIGNMENT Error (non-TypeError) at z_loc={z_local_idx}: {type(e_assign).__name__} - {e_assign}", log_queue, "CRITICAL")
                        raise

                elif log_queue and src_h <= 0:
                     log_message_gui(f"    z_loc={z_local_idx}: Skipped assignment because src_h ({src_h}) <= 0.", log_queue, "DEBUG")
                elif log_queue: # src_h != dest_h
                     log_message_gui(f"    z_loc={z_local_idx}: Skipped assignment because src_h ({src_h}) != dest_h ({dest_h}). This should be rare.", log_queue, "WARNING")
            elif log_queue and z_local_idx < 5 : # One of the clipped ranges was invalid
                log_message_gui(f"    z_loc={z_local_idx}: Skipped assignment due to invalid clipped range "
                                f"(y_s_d_c={y_s_dest_c} < y_e_d_c={y_e_dest_c} is {y_s_dest_c < y_e_dest_c}, "
                                f"y_s_s_c={y_s_src_c} < y_e_s_c={y_e_src_c} is {y_s_src_c < y_e_src_c}).", log_queue, "DEBUG")


        result_cpu = cp.asnumpy(shear_image_gpu)
        if log_queue: log_message_gui(f"  gpu_shear_image EXIT: Successfully processed chunk. Output shape {result_cpu.shape}", log_queue, "DEBUG")
        return result_cpu

    except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError, AttributeError, TypeError, ValueError) as e: # Added ValueError
        # This will catch TypeErrors from cp.asarray, cp.zeros, or re-raised from assignment
        if log_queue: log_message_gui(f"  GPU Shear Main Error (chunk_offset={z_chunk_offset}): {type(e).__name__} - {e}", log_queue, "ERROR")
        # Adding traceback for these errors from within gpu_shear_image can be very helpful
        if log_queue: log_message_gui(traceback.format_exc(), log_queue, "ERROR")
        raise # Re-raise to be handled by the caller (perform_deskewing)
    finally:
        if image_gpu is not None: del image_gpu; image_gpu = None
        if shear_image_gpu is not None: del shear_image_gpu; shear_image_gpu = None
        clear_gpu_memory()

def gpu_max_projection(image_stack, axis, log_queue=None): # Added log_queue
    """
    GPU-accelerated maximum projection along specified axis.
    Raises CuPy errors if GPU operation fails.
    """
    image_gpu = None
    mip_gpu = None
    try:
        # log_message_gui("GPU MIP: Transferring image_stack to GPU...", log_queue) # Optional
        image_gpu = cp.asarray(image_stack)
        # log_message_gui("GPU MIP: Performing cp.max on GPU...", log_queue) # Optional
        mip_gpu = cp.max(image_gpu, axis=axis)
        # log_message_gui("GPU MIP: Transferring result back to CPU...", log_queue) # Optional
        result = cp.asnumpy(mip_gpu)
        # log_message_gui("GPU MIP: Successfully completed.", log_queue) # Optional
        return result

    except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError) as e:
        # log_message_gui(f"GPU MIP: CUDA Error - {type(e).__name__}: {e}. Re-raising for CPU fallback.", log_queue, "ERROR") # Optional
        raise # Re-raise the caught CuPy/CUDA error

    finally:
        # log_message_gui("GPU MIP: Finalizing - deleting GPU arrays and clearing memory.", log_queue) # Optional
        del image_gpu
        del mip_gpu
        clear_gpu_memory()
        
        
# --- Helper for Chunked GPU Processing ---
def process_in_chunks_gpu(input_array_cpu, gpu_function, chunk_axis, num_chunks,
                          log_queue=None, fallback_to_cpu_func=None, **kwargs):
    if input_array_cpu is None:
        log_message_gui(f"Chunked GPU: Input array is None. Cannot process.", log_queue, "ERROR")
        raise ValueError("Input array to process_in_chunks_gpu cannot be None.")
    if input_array_cpu.size == 0:
        log_message_gui(f"Chunked GPU: Input array is empty. Returning copy.", log_queue, "INFO")
        # Try to apply the function to an empty array to get the correct empty output shape/dtype
        # This is complex as GPU/CPU functions might behave differently with empty inputs.
        # A safer bet for empty is to return a modified copy or have the caller handle empty input.
        # For now, let's try passing it through the CPU func if available, else copy.
        if fallback_to_cpu_func:
            try: return fallback_to_cpu_func(input_array_cpu.copy(), **kwargs), True # True as CPU func "succeeded"
            except: return input_array_cpu.copy(), True # If CPU func also fails on empty, just copy
        return input_array_cpu.copy(), True 

    if num_chunks <= 0: num_chunks = 1
    array_shape = input_array_cpu.shape
    if chunk_axis >= len(array_shape) or array_shape[chunk_axis] == 0 : # if axis size is 0, treat as 1 chunk
        num_chunks = 1
    
    min_slices_per_chunk = 1
    if num_chunks > 1 and array_shape[chunk_axis] / num_chunks < min_slices_per_chunk :
        num_chunks = max(1, int(array_shape[chunk_axis] / min_slices_per_chunk))
        log_message_gui(f"Chunked GPU: Adjusted num_chunks to {num_chunks}.", log_queue, "INFO")

    processed_chunks_cpu, all_gpu_chunks_succeeded = [], True
    log_message_gui(f"Chunked GPU: Using {num_chunks} chunk(s) along axis {chunk_axis} for {gpu_function.__name__}.", log_queue)
    
    chunk_indices = np.array_split(np.arange(array_shape[chunk_axis]), num_chunks)
    for i, indices_in_chunk_axis in enumerate(chunk_indices):
        if not indices_in_chunk_axis.size: continue
        chunk_start, chunk_end = indices_in_chunk_axis[0], indices_in_chunk_axis[-1] + 1
        slicer = [slice(None)] * input_array_cpu.ndim; slicer[chunk_axis] = slice(chunk_start, chunk_end)
        current_chunk_cpu = input_array_cpu[tuple(slicer)]

        log_message_gui(f"  Chunk {i+1}/{num_chunks} (axis {chunk_axis}: {chunk_start}-{chunk_end-1}) -> GPU", log_queue)
        chunk_gpu, processed_chunk_gpu = None, None
        try:
            if current_chunk_cpu.size == 0: # Handle empty chunk explicitly
                log_message_gui(f"  Chunk {i+1} is empty, processing accordingly.", log_queue, "INFO")
                # Try to process empty chunk with GPU func to get correct empty shape
                # This might need a dummy non-empty array of same ndim to get function signature
                # For now, assume gpu_function can handle empty or specific logic inside it handles.
                # A safer way: if gpu_function is zoom or rotate, they often handle empty.
                chunk_gpu = cp.asarray(current_chunk_cpu) # cp.asarray of empty np array is fine
                processed_chunk_gpu = gpu_function(chunk_gpu, **kwargs)

            else: # Non-empty chunk
                chunk_gpu = cp.asarray(current_chunk_cpu)
                processed_chunk_gpu = gpu_function(chunk_gpu, **kwargs)
            
            processed_chunks_cpu.append(cp.asnumpy(processed_chunk_gpu))
        except Exception as e_gpu_chunk:
            log_message_gui(f"  Chunk {i+1} GPU FAILED: {type(e_gpu_chunk).__name__} - {e_gpu_chunk}", log_queue, "ERROR")
            all_gpu_chunks_succeeded = False; break
        finally:
            if chunk_gpu is not None: del chunk_gpu; chunk_gpu = None
            if processed_chunk_gpu is not None: del processed_chunk_gpu; processed_chunk_gpu = None
            clear_gpu_memory()

    if all_gpu_chunks_succeeded and processed_chunks_cpu:
        log_message_gui("Chunked GPU: All chunks GPU OK. Stitching...", log_queue)
        try:
            # Ensure all chunks have compatible dimensions for concatenation, esp. after reshape=True in rotate
            if len(processed_chunks_cpu) > 1 and gpu_function is ndi_cp.rotate and kwargs.get('reshape', False):
                # Check if shapes (excluding chunk_axis) are consistent
                ref_shape = list(processed_chunks_cpu[0].shape)
                del ref_shape[chunk_axis]
                for ch_idx in range(1, len(processed_chunks_cpu)):
                    current_ch_shape = list(processed_chunks_cpu[ch_idx].shape)
                    del current_ch_shape[chunk_axis]
                    if current_ch_shape != ref_shape:
                        log_message_gui(f"Chunked GPU: Inconsistent shapes for rotated chunks for concat. Chunk 0: {processed_chunks_cpu[0].shape}, Chunk {ch_idx}: {processed_chunks_cpu[ch_idx].shape}. Fallback.", log_queue, "ERROR")
                        all_gpu_chunks_succeeded = False # Force fallback
                        break
            if not all_gpu_chunks_succeeded: # Recheck after shape validation
                 pass # Will fall to CPU fallback section
            elif not processed_chunks_cpu: # No chunks processed (e.g. input was empty along chunk axis)
                log_message_gui("Chunked GPU: No chunks to concatenate. Returning copy of input.", log_queue, "INFO")
                return input_array_cpu.copy(), True
            else:
                result_array_cpu = np.concatenate(processed_chunks_cpu, axis=chunk_axis)
                log_message_gui(f"Chunked GPU: Stitched shape: {result_array_cpu.shape}", log_queue)
                return result_array_cpu, True
        except ValueError as e_concat:
            log_message_gui(f"Chunked GPU: Concat FAILED: {e_concat}. GPU considered failed.", log_queue, "CRITICAL")
            all_gpu_chunks_succeeded = False # Ensure this flag is set

    # Fallback if GPU failed or concatenation failed
    if fallback_to_cpu_func:
        log_message_gui("Chunked GPU: Falling back to CPU for entire operation.", log_queue, "WARNING")
        try:
            # Ensure kwargs passed to CPU func are compatible (e.g. scipy funcs don't take cupy arrays)
            # The input_array_cpu is already CPU, so that's fine.
            return fallback_to_cpu_func(input_array_cpu, **kwargs), False 
        except Exception as e_cpu_fallback:
            log_message_gui(f"Chunked GPU: CPU fallback FAILED: {type(e_cpu_fallback).__name__} - {e_cpu_fallback}", log_queue, "ERROR")
            raise 
    else:
        raise RuntimeError("Chunked GPU processing failed and no CPU fallback provided.")

# --- GUI HELPER: Logging ---

def log_message_gui(message, log_queue=None, level="INFO"):
    formatted_message = f"[{time.strftime('%H:%M:%S')}] [{level}] {message}"
    if log_queue:
        log_queue.put(formatted_message + "\n")
    else: # Fallback if no queue (e.g. testing outside GUI)
        print(formatted_message)


# --- Core Analysis Functions ---



def generate_and_save_mips_for_gui(image_stack_yxz, output_folder_str, base_filename, 
                                   show_matplotlib_plots_flag, log_queue=None):
    log_message_gui("--- Generating and Saving Maximum Intensity Projections (MIPs) ---", log_queue)
    output_folder = Path(output_folder_str) # Work with Path objects
    base_filename_noext = Path(base_filename).stem # Use Path object's stem
    
    combined_mip_tiff_path_for_gui_display = None

    if image_stack_yxz is None or image_stack_yxz.ndim != 3 or image_stack_yxz.size == 0: # Check .size for empty
        log_message_gui("Cannot generate MIPs. Input stack not 3D or is empty.", log_queue, "WARNING")
        return None
    
    # Assuming input image_stack_yxz is (Y', X, Z') based on your log
    # Y' = y_prime_dim, X = x_dim, Z' = z_prime_dim
    y_prime_dim, x_dim, z_prime_dim = image_stack_yxz.shape
    log_message_gui(f"Input stack for MIPs (Y',X,Z'): {(y_prime_dim, x_dim, z_prime_dim)}, dtype: {image_stack_yxz.dtype}", log_queue)
    
    canvas_dtype = image_stack_yxz.dtype
    display_min_val, display_max_val = np.inf, -np.inf # Initialize correctly

    # --- Calculate MIPs ---
    mip_xy, mip_xz_orig, mip_yz_orig = None, None, None
    try:
        mip_xy = np.max(image_stack_yxz, axis=2) # Shape: (Y', X)
        if mip_xy.size > 0: 
            display_min_val = min(display_min_val, np.min(mip_xy))
            display_max_val = max(display_max_val, np.max(mip_xy))
    except Exception as e: log_message_gui(f"Error XY MIP: {e}",log_queue,"ERROR")

    try:
        mip_xz_orig = np.max(image_stack_yxz, axis=0) # Shape: (X, Z')
        if mip_xz_orig.size > 0:
            display_min_val = min(display_min_val, np.min(mip_xz_orig))
            display_max_val = max(display_max_val, np.max(mip_xz_orig))
    except Exception as e: log_message_gui(f"Error XZ MIP: {e}",log_queue,"ERROR")

    try:
        mip_yz_orig = np.max(image_stack_yxz, axis=1) # Shape: (Y', Z')
        if mip_yz_orig.size > 0:
            display_min_val = min(display_min_val, np.min(mip_yz_orig))
            display_max_val = max(display_max_val, np.max(mip_yz_orig))
    except Exception as e: log_message_gui(f"Error YZ MIP: {e}",log_queue,"ERROR")

    # --- Prepare MIPs for Display Orientation ---
    # XY: (Y' vertical, X horizontal) - already (Y', X) - GOOD
    # XZ: We want Z' vertical, X horizontal. mip_xz_orig is (X, Z'). Transpose.
    mip_xz_disp = mip_xz_orig.T if mip_xz_orig is not None else None # Shape: (Z', X)
    
    # YZ: We want Z' vertical, Y' horizontal. mip_yz_orig is (Y', Z'). Transpose.
    mip_yz_disp = mip_yz_orig.T if mip_yz_orig is not None else None # Shape: (Z', Y')

    log_message_gui("Saving individual MIP TIFFs (original orientation)...", log_queue)
    # Save original MIPs (or display-oriented ones, be consistent)
    # Let's save them in their "natural" projection orientation first
    # then save the display-oriented ones if different for the combined plot.
    for mip_data, plane_name in [(mip_xy, "XY"), (mip_xz_orig, "XZ_native"), (mip_yz_orig, "YZ_native")]:
        if mip_data is not None and mip_data.size > 0:
            save_path = output_folder / f"{base_filename_noext}_deskew_mip_{plane_name}.tif"
            try:
                # Cast to uint16 for ImageJ compatibility, consider original dtype if always float
                save_mip = mip_data
                if not np.issubdtype(save_mip.dtype, np.unsignedinteger): # If not already uint
                    if np.issubdtype(save_mip.dtype, np.floating):
                        save_mip = rescale_intensity(save_mip, out_range=(0, 65535)).astype(np.uint16)
                    elif save_mip.min() >= 0: # Signed int, positive
                        save_mip = np.clip(save_mip, 0, 65535).astype(np.uint16)
                    else: # Signed int with negatives, or other complex type - save as float32
                        save_mip = save_mip.astype(np.float32)

                tifffile.imwrite(save_path, save_mip, imagej=True)
                log_message_gui(f"Saved {plane_name} MIP to: {save_path}", log_queue)
            except Exception as e_tif:
                log_message_gui(f"Error saving {plane_name} MIP: {e_tif}", log_queue, "ERROR")

    # Check if we have all required display MIPs for combined image
    if any(x is None for x in [mip_xy, mip_xz_disp, mip_yz_disp]):
        log_message_gui("Skipping combined MIP: one or more display MIPs are invalid/empty.", log_queue, "WARNING")
        return None

    # --- Dimensions for Combined Canvas ---
    h_xy, w_xy = mip_xy.shape         # (Y', X)
    h_xz_d, w_xz_d = mip_xz_disp.shape # (Z', X)
    h_yz_d, w_yz_d = mip_yz_disp.shape # (Z', Y')

    # Sanity check dimensions based on common axes
    # Z' from mip_xz_disp and mip_yz_disp should match: h_xz_d == h_yz_d
    if h_xz_d != h_yz_d:
        log_message_gui(f"Warning: Z' dimension mismatch for combined MIP! XZ_disp height ({h_xz_d}) != YZ_disp height ({h_yz_d}). Check logic.", log_queue, "WARNING")
        # Could attempt to pad/crop, or just proceed and see. For now, proceed.
    # X from mip_xy and mip_xz_disp should match: w_xy == w_xz_d
    if w_xy != w_xz_d:
        log_message_gui(f"Warning: X dimension mismatch for combined MIP! XY width ({w_xy}) != XZ_disp width ({w_xz_d}). Check logic.", log_queue, "WARNING")
    # Y' from mip_xy and mip_yz_disp (as w_yz_d) should match: h_xy == w_yz_d
    if h_xy != w_yz_d:
        log_message_gui(f"Warning: Y' dimension mismatch for combined MIP! XY height ({h_xy}) != YZ_disp width ({w_yz_d}). Check logic.", log_queue, "WARNING")


    # Canvas layout:
    # Top row: XZ_disp (Z'v, Xh) | YZ_disp (Z'v, Y'h)
    # Bottom row: XY (Y'v, Xh) | Empty/Colorbar
    
    # Width of top row: w_xz_d (X from XZ_disp) + w_yz_d (Y' from YZ_disp)
    # Width of bottom row: w_xy (X from XY)
    total_width = max(w_xz_d + w_yz_d, w_xy)
    
    # Height of top row: h_xz_d (Z' from XZ_disp, should be same as h_yz_d)
    # Height of bottom row: h_xy (Y' from XY)
    total_height = h_xz_d + h_xy

    # Background value for canvas
    # Use a value slightly less than min if min is positive, or 0.
    # Handle cases where display_min_val might not have been updated (all MIPs empty)
    bg_value = 0
    if display_min_val != np.inf and display_min_val != -np.inf : # Check if updated
        if np.issubdtype(canvas_dtype, np.integer):
            bg_value = int(max(0, display_min_val - (display_max_val - display_min_val)*0.05)) if display_max_val > display_min_val else 0
        else: # Float
            bg_value = max(0.0, display_min_val - (display_max_val - display_min_val)*0.05) if display_max_val > display_min_val else 0.0
    if display_min_val == np.inf: # All images were empty or error
        display_min_val = 0
        display_max_val = 1 # Avoid division by zero or vmin=vmax issues
    
    combined_mip_array = np.full((total_height, total_width), fill_value=bg_value, dtype=canvas_dtype)

    try:
        # Place XZ_disp (Z' vertical, X horizontal)
        combined_mip_array[0:h_xz_d, 0:w_xz_d] = mip_xz_disp
        
        # Place YZ_disp (Z' vertical, Y' horizontal) next to XZ_disp
        # Ensure its Z' dimension (h_yz_d) matches XZ_disp's Z' dimension (h_xz_d)
        # And its Y' dimension (w_yz_d) fits
        if h_yz_d <= h_xz_d : # Common case if Z' matches
             combined_mip_array[0:h_yz_d, w_xz_d : w_xz_d + w_yz_d] = mip_yz_disp
        else: # YZ_disp Z' is taller, place and log warning or crop
            log_message_gui(f"Warning: YZ_disp Z' height ({h_yz_d}) > XZ_disp Z' height ({h_xz_d}). Cropping YZ_disp for combined view.", log_queue, "WARNING")
            combined_mip_array[0:h_xz_d, w_xz_d : w_xz_d + w_yz_d] = mip_yz_disp[:h_xz_d, :]


        # Place XY (Y' vertical, X horizontal)
        # Starts below the XZ_disp (height h_xz_d)
        combined_mip_array[h_xz_d : h_xz_d + h_xy, 0:w_xy] = mip_xy

    except Exception as e_paste:
        log_message_gui(f"Error pasting MIPs onto canvas: {e_paste}", log_queue, "ERROR")
        log_message_gui(traceback.format_exc(), log_queue, "DEBUG")
        return None

    # Save combined MIP TIFF for GUI display
    combined_mip_tiff_path_for_gui_display = output_folder / f"{base_filename_noext}_deskew_mip_COMBINED_ORTHO.tif"
    try:
        save_array_for_display = combined_mip_array
        # Type casting logic for saving (same as your original, good for ImageJ)
        if not np.issubdtype(save_array_for_display.dtype, np.floating):
            if save_array_for_display.dtype != np.uint16 and save_array_for_display.dtype != np.int16: 
                log_message_gui(f"Combined MIP dtype {save_array_for_display.dtype}, casting to uint16 for TIFF.", log_queue)
                try:
                    # Attempt to preserve range if possible, otherwise rescale
                    if save_array_for_display.min() >= 0 and save_array_for_display.max() <= 65535:
                         save_array_for_display = save_array_for_display.astype(np.uint16)
                    else: # Rescale if out of uint16 range or negative
                         save_array_for_display = rescale_intensity(save_array_for_display, out_range=(0,65535)).astype(np.uint16)
                except Exception as e_cast:
                    log_message_gui(f"Casting to uint16 failed ({e_cast}), saving as float32.", log_queue, "WARNING")
                    save_array_for_display = save_array_for_display.astype(np.float32)
        
        tifffile.imwrite(combined_mip_tiff_path_for_gui_display, save_array_for_display, imagej=True)
        log_message_gui(f"Saved COMBINED Ortho MIP TIFF to: {combined_mip_tiff_path_for_gui_display}", log_queue)
    except Exception as e_save_comb:
        log_message_gui(f"Error saving combined ortho MIP TIFF: {e_save_comb}", log_queue, "ERROR")
        combined_mip_tiff_path_for_gui_display = None # Don't return bad path

    # --- Matplotlib Visualization ---
    if show_matplotlib_plots_flag:
        plt.style.use('default') # Or 'dark_background' for different look
        
        # Determine figure size based on combined array, aim for roughly constant pixel density
        # Example: Target 100 pixels per inch for the plot display
        dpi_target = 100 
        fig_width_inches = total_width / dpi_target
        fig_height_inches = total_height / dpi_target
        # Add some margin for title and colorbar
        fig_height_inches += 1.0 
        fig_width_inches = max(6.0, fig_width_inches) # Min width
        fig_height_inches = max(5.0, fig_height_inches) # Min height
        fig_width_inches = min(15.0, fig_width_inches) # Max width
        fig_height_inches = min(12.0, fig_height_inches) # Max height


        fig_comb, ax_comb = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
        
        current_display_vmin_plot = display_min_val if display_min_val != np.inf else 0
        current_display_vmax_plot = display_max_val if display_max_val != -np.inf else (current_display_vmin_plot + 1)
        if current_display_vmax_plot <= current_display_vmin_plot: 
            current_display_vmax_plot = current_display_vmin_plot + (1 if current_display_vmin_plot < np.iinfo(save_array_for_display.dtype).max else 0.001)


        im_comb = ax_comb.imshow(combined_mip_array, cmap='gray', 
                                 vmin=current_display_vmin_plot, vmax=current_display_vmax_plot,
                                 interpolation='nearest', aspect='equal', origin='upper') # Origin 'upper' is standard for imshow

        ax_comb.set_title(f'Combined Ortho MIPs: {base_filename_noext}', fontsize=10, pad=15)
        ax_comb.axis('off')

        # Add text labels for each view. Positions are relative to (0,0) at top-left.
        # Use a small offset from the edges.
        text_offset = 5 # pixels
        text_props = dict(facecolor='black', alpha=0.6, pad=2, edgecolor='yellow')
        font_props = dict(color='yellow', fontsize=8, ha='left', va='top')

        # XZ_disp (Top-Left)
        ax_comb.text(text_offset, text_offset, 'XZ (Z\' vert)', **font_props, bbox=text_props)
        # YZ_disp (Top-Right)
        ax_comb.text(w_xz_d + text_offset, text_offset, 'YZ (Z\' vert)', **font_props, bbox=text_props)
        # XY (Bottom-Left)
        ax_comb.text(text_offset, h_xz_d + text_offset, 'XY (Y\' vert)', **font_props, bbox=text_props)
        
        # Add lines to delineate the subplots (optional)
        ax_comb.axhline(y=h_xz_d - 0.5, color='gray', linestyle='--', linewidth=0.5, xmin=0, xmax=float(w_xz_d + w_yz_d)/total_width if total_width > 0 else 1)
        ax_comb.axvline(x=w_xz_d - 0.5, color='gray', linestyle='--', linewidth=0.5, ymin=1 - float(h_xz_d)/total_height if total_height > 0 else 0, ymax=1) # ymin relative to top

        # Add a colorbar
        # Position the colorbar to the right of the XY plot if there's space, or below all.
        # For simplicity, let's add it to the side of the whole figure.
        cbar = fig_comb.colorbar(im_comb, ax=ax_comb, shrink=0.7, aspect=20, pad=0.02, location='right')
        cbar.set_label('Max Intensity', size=9)
        cbar.ax.tick_params(labelsize=8)
        
        fig_comb.tight_layout(rect=[0, 0, 0.95, 0.95]) # Adjust rect to make space for colorbar and title
        
        plot_output_dir_viz = output_folder / "analysis_plots" # Keep consistent name
        plot_output_dir_viz.mkdir(parents=True, exist_ok=True)
        save_path_combined_png = plot_output_dir_viz / f"{base_filename_noext}_deskew_mip_COMBINED_ORTHO_viz.png"
        try:
            plt.savefig(save_path_combined_png, dpi=150, bbox_inches='tight')
            log_message_gui(f"Saved COMBINED Ortho MIP PNG (Viz) to: {save_path_combined_png}", log_queue)
        except Exception as e_save_png:
            log_message_gui(f"Error saving combined ortho MIP PNG: {e_save_png}", log_queue, "ERROR")
        finally:
            plt.close(fig_comb) # Explicitly close plot

    return str(combined_mip_tiff_path_for_gui_display) if combined_mip_tiff_path_for_gui_display else None
def matlab_round(x):
    """Mimics MATLAB's round-away-from-zero for .5 cases."""
    return np.floor(x + 0.5) if x >= 0 else np.ceil(x - 0.5)

def perform_deskewing(full_file_path: str, dx_um: float, dz_um: float, angle_deg: float, flip_direction: int,
                      save_intermediate_shear: bool, show_deskew_plots: bool, log_queue=None,
                      num_z_chunks_for_gpu_shear: int = 4, 
                      min_z_slices_per_chunk_gpu_shear: int = 32, 
                      gpu_shear_fallback_to_cpu_process: bool = True,
                      num_x_chunks_for_gpu_zoom_rotate: int = 4, 
                      gpu_zoom_rotate_fallback_to_cpu: bool = True,
                      # --- NEW PARAMETERS FOR SMOOTHING ---
                      apply_post_shear_smoothing: bool = False, # Default to False
                      smoothing_sigma_yc: float = 0.7, # Sigma for Y'c axis (shear direction after rotation)
                      smoothing_sigma_x: float = 0.0,  # Sigma for X axis (usually no blur here)
                      smoothing_sigma_zc: float = 0.0,  # Sigma for Z'c axis (usually no blur here)
                      save_final_deskew: bool = True
                      ):
    log_message_gui("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", log_queue)
    log_message_gui("%           Deskew & Rotate Light-Sheet Data (Python Version)                      %", log_queue)
    log_message_gui("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%", log_queue)

    start_time_total = time.time()
    output_path_top_deskewed_tiff_str=None; final_dx_eff_um=dx_um; final_dz_eff_um=dx_um 
    output_folder_top_str=None; processed_file_name_str=None; combined_mip_display_path_str=None
    image_stack=None; shear_image=None; scaled_shear_image_cpu=None; scaled_mip_yz_cpu=None
    rotated_scaled_mip=None; rotated_stack_cpu=None; cropped_rotate_stack_cpu=None
    rot_top_shear_image=None; final_image_to_process=None; final_image_for_mips_and_save=None
    final_image_to_save_astype=None; final_image_to_save_zyx=None
    zsize_full = 0 # Initialize in case of early exit

    try:
        full_path_obj = Path(full_file_path)
        file_path_obj, actual_file_name_str, actual_file_stem_str = full_path_obj.parent, full_path_obj.name, full_path_obj.stem
        processed_file_name_str = actual_file_name_str
        log_message_gui(f'Processing file for deskewing: {full_file_path}', log_queue)
        top_output_folder_name = f'Top_shear{angle_deg}_py_{actual_file_stem_str}' 
        output_folder_top = file_path_obj / top_output_folder_name
        output_folder_top.mkdir(parents=True, exist_ok=True); output_folder_top_str = str(output_folder_top)

        log_message_gui("\n--- Reading and Preparing Image ---", log_queue); tic_read = time.time()
        with tifffile.TiffFile(full_file_path) as tif: image_stack_raw = tif.asarray()
        if image_stack_raw.ndim<3: raise ValueError("Input < 3D.")
        if image_stack_raw.ndim>3:
            if image_stack_raw.shape[0]==1 and image_stack_raw.ndim==4: image_stack_raw=np.squeeze(image_stack_raw,axis=0)
            if image_stack_raw.ndim>3: log_message_gui(f"Warn:Input >3D {image_stack_raw.shape}. Using [0,...].",log_queue,"WARNING"); image_stack_raw=image_stack_raw[0,...]
        if image_stack_raw.ndim!=3: raise ValueError(f"Not 3D. ndim:{image_stack_raw.ndim}")
        s_in=image_stack_raw.shape; z_thr=0.8
        if s_in[0]<z_thr*s_in[1] and s_in[0]<z_thr*s_in[2]: image_stack=image_stack_raw.transpose(1,2,0)
        elif s_in[2]<z_thr*s_in[0] and s_in[2]<z_thr*s_in[1]: image_stack=image_stack_raw
        else: log_message_gui(f"Warn:Input {s_in} ambiguous. Assuming YXZ.",log_queue,"WARNING"); image_stack=image_stack_raw
        del image_stack_raw; gc.collect(); toc_read=time.time()
        log_message_gui(f"Read in {toc_read-tic_read:.2f}s. YXZ:{image_stack.shape},{image_stack.dtype}",log_queue)
        out_dtype=np.uint16
        if image_stack.ndim!=3 or min(image_stack.shape)<=0 and image_stack.size > 0 : raise ValueError("Invalid dims post-prep.") # Allow empty if size is 0

        ysize, xsize, zsize_full = image_stack.shape
        new_dz_s = dz_um*math.cos(math.radians(angle_deg))
        log_message_gui("\n--- Applying Shear Transformation ---",log_queue); tic_s=time.time()
        cz_ref,px_s_z = math.floor(zsize_full/2),new_dz_s/dx_um
        max_y_off = int(max(abs(matlab_round(flip_direction*(0-cz_ref)*px_s_z)), 
                               abs(matlab_round(flip_direction*((zsize_full-1)-cz_ref)*px_s_z))))
        pad_Y = ysize+2*max_y_off
        log_message_gui(f"Shear:cz_ref={cz_ref},px_s_z={px_s_z:.4f},max_y_off={max_y_off},pad_Y={pad_Y}",log_queue)
        if pad_Y<=0 or xsize<=0 and zsize_full > 0 : raise ValueError("Invalid calc dims for shear.") # Allow xsize=0 if zsize_full=0

        gpu_s_ok=False
        eff_z_chunks_s=num_z_chunks_for_gpu_shear
        if zsize_full > 0 and eff_z_chunks_s > 0 and zsize_full/eff_z_chunks_s < min_z_slices_per_chunk_gpu_shear:
            eff_z_chunks_s = max(1, int(zsize_full/min_z_slices_per_chunk_gpu_shear))
            if eff_z_chunks_s > num_z_chunks_for_gpu_shear and num_z_chunks_for_gpu_shear >=1 : eff_z_chunks_s = num_z_chunks_for_gpu_shear
        if eff_z_chunks_s == 0 : eff_z_chunks_s = 1
        log_message_gui(f"Attempting GPU shear: {eff_z_chunks_s} Z-chunk(s).",log_queue)
        
        s_chunks_cpu=[]; all_gpu_s_chunks_ok=True
        if zsize_full > 0:
            z_idx_chunks_s=np.array_split(np.arange(zsize_full),eff_z_chunks_s)
            for i, ch_z_idx_s in enumerate(z_idx_chunks_s):
                if not ch_z_idx_s.size: continue
                z_s,z_e=ch_z_idx_s[0],ch_z_idx_s[-1]+1
                log_message_gui(f" GPU Shear Z-chunk {i+1}/{eff_z_chunks_s} (Z:{z_s}-{z_e-1})...",log_queue)
                try:
                    s_ch_cpu = gpu_shear_image(image_stack[:,:,z_s:z_e],zsize_full,cz_ref,px_s_z,flip_direction,max_y_off,z_s,log_queue)
                    s_chunks_cpu.append(s_ch_cpu)
                except Exception as e_gpu_s_ch:
                    log_message_gui(f" GPU shear chunk {i+1} FAILED:{type(e_gpu_s_ch).__name__}.",log_queue,"ERROR")
                    all_gpu_s_chunks_ok=False
                    if not gpu_shear_fallback_to_cpu_process: raise RuntimeError("GPU shear chunk failed, no fallback.") from e_gpu_s_ch
                    break
            if all_gpu_s_chunks_ok and s_chunks_cpu:
                try: shear_image=np.concatenate(s_chunks_cpu,axis=2); gpu_s_ok=True
                except ValueError as e_cat: log_message_gui(f"Concat sheared chunks FAILED:{e_cat}.Fallback CPU.",log_queue,"CRITICAL"); gpu_s_ok=False
                del s_chunks_cpu; gc.collect()
            else: gpu_s_ok=False
        else: shear_image=np.zeros((pad_Y,xsize,0),dtype=image_stack.dtype); gpu_s_ok=True # Empty Z
        
        if not gpu_s_ok:
            if not gpu_shear_fallback_to_cpu_process and zsize_full > 0: raise RuntimeError("GPU shear failed, no fallback.")
            log_message_gui("Fallback to CPU for shear.",log_queue)
            shear_image=np.zeros((pad_Y,xsize,zsize_full),dtype=image_stack.dtype)
            if zsize_full > 0:
                for z_idx in range(zsize_full):
                    y_off=int(matlab_round(flip_direction*(z_idx-cz_ref)*px_s_z))
                    y_s_d,y_e_d=y_off+max_y_off,y_off+max_y_off+ysize
                    y_s_d_c,y_e_d_c=max(0,y_s_d),min(pad_Y,y_e_d)
                    y_s_s_c,y_e_s_c=max(0,-y_s_d),ysize-max(0,y_e_d-pad_Y)
                    if y_s_d_c<y_e_d_c and y_s_s_c<y_e_s_c and (y_e_s_c-y_s_s_c)==(y_e_d_c-y_s_d_c)>0:
                        shear_image[y_s_d_c:y_e_d_c,:,z_idx]=image_stack[y_s_s_c:y_e_s_c,:,z_idx]
        del image_stack; image_stack=None; gc.collect()
        if shear_image is None: raise RuntimeError("Shear image None post-shear.")
        toc_s=time.time(); log_message_gui(f"Shear done {toc_s-tic_s:.2f}s. Shape:{shear_image.shape}",log_queue)
        
        if save_intermediate_shear:
            if shear_image.nbytes/(1024**3)<4.0:
                log_message_gui("Saving intermediate shear...",log_queue)
                s_pth=output_folder_top/f"INTERMEDIATE_{actual_file_stem_str}_sheared.tif"; s_sv=shear_image
                if s_sv.dtype!=out_dtype:s_sv=np.clip(s_sv.copy() if s_sv is shear_image else s_sv,np.iinfo(out_dtype).min,np.iinfo(out_dtype).max).astype(out_dtype)
                tifffile.imwrite(str(s_pth),s_sv.transpose(2,0,1),imagej=True,metadata={'axes':'ZYX'},bigtiff=(s_sv.nbytes/(1024**3)>3.9))
                if s_sv is not shear_image: del s_sv
                log_message_gui(f"Intermediate saved: {s_pth}", log_queue)
            else: log_message_gui(f"Intermediate shear ({shear_image.nbytes/(1024**3):.2f}GB)>=4GB,skip.",log_queue,"WARNING")

        log_message_gui("\n--- Rotation to Top View ---",log_queue); tic_rot_s=time.time()
        scl_z_ax = abs(dz_um*math.sin(math.radians(angle_deg))/dx_um)
        log_message_gui(f"Rotation: Z scaling factor={scl_z_ax:.4f}",log_queue)

        log_message_gui("Rotation: Z-scaling (Zoom)...",log_queue)
        zoom_p={'zoom':(1.0,1.0,scl_z_ax),'order':1,'mode':'constant','cval':0.0,'prefilter':True}
        scaled_shear_image_cpu,zoom_gpu_ok = process_in_chunks_gpu(shear_image,ndi_cp.zoom,1,num_x_chunks_for_gpu_zoom_rotate,log_queue,scipy_zoom if gpu_zoom_rotate_fallback_to_cpu else None,**zoom_p)
        if not zoom_gpu_ok and not gpu_zoom_rotate_fallback_to_cpu: raise RuntimeError("GPU Zoom failed, no fallback.")
        log_msg = "GPU (chunked)" if zoom_gpu_ok else "CPU (fallback)"
        log_message_gui(f"Rotation: Z-scaling (Zoom) done on {log_msg}.",log_queue)
        del shear_image; shear_image=None; gc.collect()
        if scaled_shear_image_cpu is None or scaled_shear_image_cpu.ndim!=3:
            if not (scaled_shear_image_cpu is not None and scaled_shear_image_cpu.ndim==3 and scaled_shear_image_cpu.shape[2]==0 and zsize_full==0): raise RuntimeError("Scaled_shear_image invalid.")
        if scaled_shear_image_cpu.size>0 and min(scaled_shear_image_cpu.shape)<=0: raise RuntimeError(f"Scaled_shear_image zero dim:{scaled_shear_image_cpu.shape}")
        log_message_gui(f"Rotation: Scaled shear shape:{scaled_shear_image_cpu.shape}",log_queue)

        log_message_gui("Rotation: Calculating BBox...",log_queue)
        min_r,max_r,min_c,max_c=0,0,0,0; rot_ang_deskew = -1*flip_direction*angle_deg
        if scaled_shear_image_cpu.size>0:
            try: scaled_mip_yz_cpu=gpu_max_projection(scaled_shear_image_cpu,1,log_queue); log_message_gui("GPU MIP for BBox OK.",log_queue)
            except Exception as e_gpu_mip: log_message_gui(f"GPU MIP FAILED ({type(e_gpu_mip).__name__}).Fallback CPU.",log_queue,"WARNING"); scaled_mip_yz_cpu=np.max(scaled_shear_image_cpu,1)
            if scaled_mip_yz_cpu is None: raise RuntimeError("scaled_mip_yz_cpu None.")
            rotated_scaled_mip=scipy_rotate(scaled_mip_yz_cpu,angle=rot_ang_deskew,reshape=True,order=1,mode='constant',cval=0,prefilter=True)
            del scaled_mip_yz_cpu; scaled_mip_yz_cpu=None; gc.collect()
            min_r,max_r,min_c,max_c=0,rotated_scaled_mip.shape[0],0,rotated_scaled_mip.shape[1]
            if np.any(rotated_scaled_mip>1e-9):rows,cols=np.where(rotated_scaled_mip>1e-9);min_r,max_r,min_c,max_c=np.min(rows),np.max(rows)+1,np.min(cols),np.max(cols)+1
            log_message_gui(f"Rotation: BBox rows [{min_r}:{max_r}], cols [{min_c}:{max_c}]",log_queue)
            del rotated_scaled_mip; rotated_scaled_mip=None; gc.collect()
        else: log_message_gui("Scaled shear empty, skip MIP for BBox.",log_queue,"INFO")

        log_message_gui(f"Rotation: Rotating stack by {rot_ang_deskew:.2f} deg...",log_queue)
        rot_p={'angle':rot_ang_deskew,'axes':(0,2),'reshape':True,'order':1,'mode':'constant','cval':0.0,'prefilter':True}
        rotated_stack_cpu,rot_gpu_ok = process_in_chunks_gpu(scaled_shear_image_cpu,ndi_cp.rotate,1,num_x_chunks_for_gpu_zoom_rotate,log_queue,scipy_rotate if gpu_zoom_rotate_fallback_to_cpu else None,**rot_p)
        if not rot_gpu_ok and not gpu_zoom_rotate_fallback_to_cpu: raise RuntimeError("GPU Rotate failed, no fallback.")
        log_msg = "GPU (chunked)" if rot_gpu_ok else "CPU (fallback)"
        log_message_gui(f"Rotation: Stack rotation done on {log_msg}.",log_queue)
        del scaled_shear_image_cpu; scaled_shear_image_cpu=None; gc.collect()
        if rotated_stack_cpu is None: raise RuntimeError("rotated_stack_cpu None.")
        if rotated_stack_cpu.size>0 and min(rotated_stack_cpu.shape)<=0: raise RuntimeError(f"Rotated stack zero dim:{rotated_stack_cpu.shape}")
        log_message_gui(f"Rotation: Rotated stack shape:{rotated_stack_cpu.shape}",log_queue)

        if rotated_stack_cpu.size>0:
            crop_r0,crop_r1=max(0,min_r),min(rotated_stack_cpu.shape[0],max_r)
            crop_c0,crop_c1=max(0,min_c),min(rotated_stack_cpu.shape[2],max_c)
            if crop_r0<crop_r1 and crop_c0<crop_c1: cropped_rotate_stack_cpu=rotated_stack_cpu[crop_r0:crop_r1,:,crop_c0:crop_c1]
            else: log_message_gui("Warn:Invalid crop.Using uncropped.",log_queue,"WARNING"); cropped_rotate_stack_cpu=rotated_stack_cpu
        else: cropped_rotate_stack_cpu=rotated_stack_cpu 
        if rotated_stack_cpu is not cropped_rotate_stack_cpu: del rotated_stack_cpu; rotated_stack_cpu=None
        gc.collect(); rot_top_shear_image=cropped_rotate_stack_cpu; cropped_rotate_stack_cpu=None
        if rot_top_shear_image is None or rot_top_shear_image.ndim!=3:
            if not (rot_top_shear_image is not None and rot_top_shear_image.ndim==3 and rot_top_shear_image.shape[2]==0 and zsize_full==0): raise RuntimeError("Invalid crop (ndim/type).")
        if rot_top_shear_image.size>0 and min(rot_top_shear_image.shape)<=0: raise RuntimeError(f"Cropped zero dim:{rot_top_shear_image.shape}")
        toc_rot_s=time.time(); log_message_gui(f"Rotation done {toc_rot_s-tic_rot_s:.2f}s. Shape:{rot_top_shear_image.shape}",log_queue)
        
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # SMOOTHING CODE BLOCK (GPU Attempt First)
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        if apply_post_shear_smoothing and rot_top_shear_image is not None and rot_top_shear_image.size > 0:
            if smoothing_sigma_yc > 0 or smoothing_sigma_x > 0 or smoothing_sigma_zc > 0 :
                log_message_gui(f"Applying post-shear smoothing (Gaussian) to Y'cX Z'c ({rot_top_shear_image.shape}) "
                                f"with sigmas: Yc'={smoothing_sigma_yc}, X={smoothing_sigma_x}, Zc'={smoothing_sigma_zc}", log_queue)
                tic_smooth = time.time()
                
                sigmas_for_filter = (smoothing_sigma_yc, smoothing_sigma_x, smoothing_sigma_zc)
                original_dtype_cpu = rot_top_shear_image.dtype # CPU array's original dtype
                
                smoothed_image_final_cpu = None # Will hold the final smoothed image on CPU
                gpu_smoothing_attempted = False
                gpu_smoothing_successful = False
                
                # --- Attempt GPU Smoothing ---
                # Declare GPU array variables outside try for finally block
                rot_top_shear_image_gpu = None
                image_to_filter_gpu = None
                smoothed_image_gpu = None

                try:
                    # Check if rot_top_shear_image can be reasonably processed on GPU
                    # This is a heuristic, adjust as needed based on your typical GPU memory
                    # Example: If image > 8GB, maybe skip GPU attempt for smoothing if memory is often an issue
                    # For now, let's always attempt if CuPy is available.
                    # You could add a size check here: if rot_top_shear_image.nbytes > SOME_THRESHOLD_BYTES: raise cp.cuda.memory.OutOfMemoryError("Image too large for GPU smoothing attempt")

                    log_message_gui("  Smoothing: Attempting GPU Gaussian filter...", log_queue, "DEBUG")
                    gpu_smoothing_attempted = True

                    rot_top_shear_image_gpu = cp.asarray(rot_top_shear_image)
                    
                    image_to_filter_gpu = rot_top_shear_image_gpu
                    # Convert to float on GPU if necessary, as gaussian_filter works best with float
                    if not cp.issubdtype(rot_top_shear_image_gpu.dtype, cp.floating):
                        log_message_gui(f"  Smoothing: Converting GPU image from {rot_top_shear_image_gpu.dtype} to cp.float32 for smoothing.", log_queue, "DEBUG")
                        image_to_filter_gpu = rot_top_shear_image_gpu.astype(cp.float32)
                    
                    smoothed_image_gpu = ndi_cp.gaussian_filter(
                        image_to_filter_gpu,
                        sigma=sigmas_for_filter, # Sigmas are for (Y'c, X, Z'c)
                        order=0,
                        mode='constant', # Check CuPy docs for supported modes ('mirror', 'reflect', 'nearest')
                        cval=0.0
                        # `prefilter` is not an explicit argument for cupyx's gaussian_filter
                    )

                    log_message_gui("  Smoothing: Transferring smoothed image from GPU to CPU...", log_queue, "DEBUG")
                    smoothed_image_float_cpu = cp.asnumpy(smoothed_image_gpu)
                    
                    # Convert back to original CPU dtype
                    if not np.issubdtype(original_dtype_cpu, np.floating):
                        log_message_gui(f"  Smoothing: Converting CPU smoothed image back to {original_dtype_cpu}.", log_queue, "DEBUG")
                        if np.issubdtype(original_dtype_cpu, np.integer):
                            min_val, max_val = np.iinfo(original_dtype_cpu).min, np.iinfo(original_dtype_cpu).max
                            smoothed_image_final_cpu = np.clip(smoothed_image_float_cpu, min_val, max_val).astype(original_dtype_cpu)
                        else: # e.g. boolean, though unlikely for images needing smoothing
                            smoothed_image_final_cpu = smoothed_image_float_cpu.astype(original_dtype_cpu)
                    else: # Original was float
                        smoothed_image_final_cpu = smoothed_image_float_cpu.astype(original_dtype_cpu) # Ensure exact original float type (e.g. float32 vs float64)
                    
                    gpu_smoothing_successful = True
                    log_message_gui("  Post-shear smoothing on GPU successful.", log_queue)

                except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError) as e_gpu_mem:
                    log_message_gui(f"  GPU smoothing FAILED (CUDA Memory/Runtime Error: {type(e_gpu_mem).__name__}). Falling back to CPU.", log_queue, "WARNING")
                    # log_message_gui(traceback.format_exc(), log_queue, "DEBUG") # Optional detailed traceback
                    gpu_smoothing_successful = False
                except (AttributeError, TypeError, ValueError) as e_gpu_op: # Catch op errors like wrong type or cupyx not found
                    log_message_gui(f"  GPU smoothing FAILED (Operation Error: {type(e_gpu_op).__name__} - {e_gpu_op}). Falling back to CPU.", log_queue, "WARNING")
                    # log_message_gui(traceback.format_exc(), log_queue, "DEBUG")
                    gpu_smoothing_successful = False
                except Exception as e_gpu_other: # Catch any other unexpected error during GPU attempt
                    log_message_gui(f"  GPU smoothing FAILED (Unexpected Error: {type(e_gpu_other).__name__} - {e_gpu_other}). Falling back to CPU.", log_queue, "ERROR")
                    log_message_gui(traceback.format_exc(), log_queue, "ERROR") # Log full traceback for unexpected
                    gpu_smoothing_successful = False
                finally:
                    # Clean up GPU arrays explicitly
                    if rot_top_shear_image_gpu is not None: del rot_top_shear_image_gpu; rot_top_shear_image_gpu = None
                    # image_to_filter_gpu is an alias or a new array if type casting happened
                    if image_to_filter_gpu is not None and image_to_filter_gpu is not rot_top_shear_image_gpu : # only del if it was a distinct copy
                        del image_to_filter_gpu
                    image_to_filter_gpu = None # Reset alias too
                    if smoothed_image_gpu is not None: del smoothed_image_gpu; smoothed_image_gpu = None
                    if gpu_smoothing_attempted: clear_gpu_memory() # Clear pool only if GPU was touched

                # --- CPU Fallback for Smoothing (if GPU failed or wasn't attempted) ---
                if not gpu_smoothing_successful:
                    log_message_gui("  Applying post-shear smoothing on CPU...", log_queue)
                    
                    image_to_filter_cpu = rot_top_shear_image # Start with the original CPU array
                    if not np.issubdtype(original_dtype_cpu, np.floating):
                        log_message_gui(f"  Smoothing: Converting CPU image from {original_dtype_cpu} to float32 for smoothing.", log_queue, "DEBUG")
                        image_to_filter_cpu = rot_top_shear_image.astype(np.float32)

                    smoothed_image_float_cpu = scipy_gaussian_filter(
                        image_to_filter_cpu, 
                        sigma=sigmas_for_filter, 
                        order=0, 
                        mode='constant', 
                        cval=0.0
                    )
                    
                    # Convert back to original CPU dtype
                    if not np.issubdtype(original_dtype_cpu, np.floating):
                        log_message_gui(f"  Smoothing: Converting CPU smoothed image back to {original_dtype_cpu}.", log_queue, "DEBUG")
                        if np.issubdtype(original_dtype_cpu, np.integer):
                            min_val, max_val = np.iinfo(original_dtype_cpu).min, np.iinfo(original_dtype_cpu).max
                            smoothed_image_final_cpu = np.clip(smoothed_image_float_cpu, min_val, max_val).astype(original_dtype_cpu)
                        else:
                            smoothed_image_final_cpu = smoothed_image_float_cpu.astype(original_dtype_cpu)
                    else: # Original was float
                        smoothed_image_final_cpu = smoothed_image_float_cpu.astype(original_dtype_cpu)
                    
                    # Clean up intermediate CPU float array if it was created and distinct
                    if image_to_filter_cpu is not rot_top_shear_image : del image_to_filter_cpu
                    # smoothed_image_float_cpu will be replaced by smoothed_image_final_cpu or is the same
                    if smoothed_image_float_cpu is not smoothed_image_final_cpu: del smoothed_image_float_cpu
                    log_message_gui("  Post-shear smoothing on CPU successful.", log_queue)


                # Update rot_top_shear_image with the smoothed version
                if smoothed_image_final_cpu is not None: # Check if smoothing actually produced an output
                    if rot_top_shear_image is not smoothed_image_final_cpu : # If a new array was made
                        del rot_top_shear_image # Delete the old (unsmoothed) one
                    rot_top_shear_image = smoothed_image_final_cpu # Assign the new smoothed one
                else: # This case should ideally not be reached if logic is correct
                    log_message_gui("  Error: smoothed_image_final_cpu is None after smoothing attempts. Original image retained.", log_queue, "ERROR")

                gc.collect()
                toc_smooth = time.time()
                log_message_gui(f"Post-shear smoothing completed in {toc_smooth - tic_smooth:.2f}s. New shape: {rot_top_shear_image.shape if rot_top_shear_image is not None else 'None'}", log_queue)
            else:
                log_message_gui("Post-shear smoothing skipped: all sigmas are zero.", log_queue, "INFO")
        elif apply_post_shear_smoothing: # but rot_top_shear_image is None or empty
            log_message_gui("Post-shear smoothing skipped: image is None or empty.", log_queue, "INFO")
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        # +++ END OF SMOOTHING CODE BLOCK +++
        # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        log_message_gui("\n--- Preparing and Saving Final View ---",log_queue)
        final_image_to_process=rot_top_shear_image
        out_gb=final_image_to_process.nbytes/(1024**3) if final_image_to_process.size>0 else 0; ds_f_z=1
        if out_gb>=12.0: ds_f_z=4
        elif out_gb>=8.0: ds_f_z=3
        elif out_gb>=4.0: ds_f_z=2
        final_dz_eff_um=dx_um*ds_f_z

        final_image_for_mips_and_save=final_image_to_process
        if ds_f_z>1 and final_image_to_process.shape[2]>1:
            log_message_gui(f"Deskew:Downsample Z' by 1/{ds_f_z}x. Size:{out_gb:.2f}GB",log_queue)
            tgt_z=max(1,round(final_image_to_process.shape[2]/ds_f_z))
            zm_f=tgt_z/final_image_to_process.shape[2]
            if zm_f<1.0: final_image_for_mips_and_save=scipy_zoom(final_image_to_process,(1.0,1.0,zm_f),order=1,mode='constant',cval=0.0,prefilter=True); log_message_gui(f"Deskew:Shape post Z downsample={final_image_for_mips_and_save.shape}",log_queue)
            else: log_message_gui(f"Deskew:Zoom factor {zm_f:.3f}>=1.0,skip Z downsample.",log_queue); ds_f_z=1;final_dz_eff_um=dx_um
        
        final_image_to_save_astype=final_image_for_mips_and_save
        if np.issubdtype(out_dtype,np.integer):
            mx_v,mn_v=np.iinfo(out_dtype).max,np.iinfo(out_dtype).min
            if final_image_for_mips_and_save.size>0 and np.isnan(final_image_for_mips_and_save).any():
                final_image_to_save_astype=np.nan_to_num(final_image_for_mips_and_save.copy() if final_image_to_save_astype is final_image_for_mips_and_save else final_image_to_save_astype,nan=0)
            if final_image_to_save_astype.dtype!=out_dtype:
                final_image_to_save_astype=np.clip(final_image_to_save_astype.copy() if final_image_to_save_astype is final_image_for_mips_and_save else final_image_to_save_astype,mn_v,mx_v).astype(out_dtype)
        elif final_image_for_mips_and_save.dtype!=out_dtype:
            final_image_to_save_astype=(final_image_for_mips_and_save.copy() if final_image_to_save_astype is final_image_for_mips_and_save else final_image_to_save_astype).astype(out_dtype)
        
        final_image_to_save_zyx=final_image_to_save_astype.transpose(2,0,1)
        log_message_gui(f"Final Save (ZYX):{final_image_to_save_zyx.shape},{final_image_to_save_zyx.dtype}",log_queue)
        
        if save_final_deskew:
            out_pth_final_tif=output_folder_top/actual_file_name_str
            tifffile.imwrite(str(out_pth_final_tif),final_image_to_save_zyx,imagej=True,metadata={'axes':'ZYX'},bigtiff=(final_image_to_save_zyx.nbytes/(1024**3)>3.9))
            output_path_top_deskewed_tiff_str=str(out_pth_final_tif)
            log_message_gui(f"Final saved: {output_path_top_deskewed_tiff_str}",log_queue)

        note_pth=output_folder_top/'deskew_note.txt'
        s_gpu_info_s="N/A"
        if 'gpu_s_ok' in locals(): s_gpu_info_s = str(locals().get('eff_z_chunks_s','GPU(Unk)')) if gpu_s_ok else "CPU Fallback"
        z_m=f"z'~{ds_f_z}*xy um" if ds_f_z>1 else "z'~xy um"
        with open(note_pth,'w') as f:
            f.write(f"--Deskew Notes--\nIn:{actual_file_name_str}\ndx:{dx_um},dz:{dz_um},ang:{angle_deg},flip:{flip_direction}\n")
            f.write(f"Shear Z-Chunks GPU:{s_gpu_info_s}\nZoom/Rot X-Chunks GPU:{num_x_chunks_for_gpu_zoom_rotate}\n")
            f.write(f"Z' Downsample:{ds_f_z}\n{z_m}\nFinal ZYX:{final_image_to_save_zyx.shape},{final_image_to_save_zyx.dtype}\n")

        combined_mip_display_path_str=generate_and_save_mips_for_gui(final_image_for_mips_and_save,str(output_folder_top),actual_file_name_str,show_deskew_plots,log_queue)

    except Exception as e_pipeline:
        log_message_gui(f"[CRITICAL_ERROR] Pipeline error: {type(e_pipeline).__name__} - {e_pipeline}",log_queue,"CRITICAL")
        log_message_gui(traceback.format_exc(),log_queue,"CRITICAL")

    finally:
        log_message_gui("Deskew: Final cleanup.",log_queue)
        vars_to_del=['final_image_to_save_zyx','final_image_to_save_astype','final_image_for_mips_and_save',
                       'final_image_to_process','rot_top_shear_image','cropped_rotate_stack_cpu','rotated_stack_cpu',
                       'scaled_mip_yz_cpu','rotated_scaled_mip','scaled_shear_image_cpu','shear_image','image_stack']
        for var_n in vars_to_del:
            if var_n in locals() and locals()[var_n] is not None:
                try: del locals()[var_n]
                except NameError: pass 
        gc.collect(); clear_gpu_memory()
        end_time_total=time.time()
        log_message_gui(f"Total deskew time: {end_time_total-start_time_total:.2f}s.",log_queue)

    return output_path_top_deskewed_tiff_str,final_dx_eff_um,final_dz_eff_um,output_folder_top_str,processed_file_name_str,combined_mip_display_path_str


# --- Decorrelation and PSF Analysis Functions ---
# [Rest of your code remains unchanged - decorrelation and PSF analysis functions]
# [GUI application class and main code also remain unchanged]
# --- Decorrelation and PSF Analysis Functions (Unchanged from previous version) ---
# --- [OMITTED FOR BREVITY] ---
DECORR_POD_SIZE = 30
DECORR_POD_ORDER = 8

def _decorr_fft(image): return fftshift(fftn(fftshift(image)))
def _decorr_ifft(im_fft): return ifftshift(ifftn(ifftshift(im_fft)))
def _decorr_masked_fft(im, mask, size): return (mask * _decorr_fft(im)).ravel()[: size // 2]

def decorr_apodise(image, border, order=DECORR_POD_ORDER):
    nx, ny = image.shape
    window_x = ss.general_gaussian(nx, order, nx // 2 - border)
    window_y = ss.general_gaussian(ny, order, ny // 2 - border)
    window = np.outer(window_x, window_y)
    return window * image

class DecorrImageDecorr:
    pod_size = DECORR_POD_SIZE
    pod_order = DECORR_POD_ORDER

    def __init__(self, image, pixel_size=1.0, square_crop=True):
        if not image.ndim == 2: raise ValueError("ImageDecorr expects a 2D image.")
        image = image.astype(np.float64)
        self.image = decorr_apodise(image, self.pod_size, self.pod_order)
        self.pixel_size = float(pixel_size)
        nx, ny = self.image.shape

        if square_crop:
            n = min(nx, ny); n = n - (1 - n % 2)
            self.image = self.image[:n, :n]; self.size = n * n
            xx, yy = np.meshgrid(np.linspace(-1, 1, n), np.linspace(-1, 1, n))
        else:
            nx = nx - (1 - nx % 2); ny = ny - (1 - ny % 2)
            self.image = self.image[:nx, :ny]; self.size = nx * ny
            xx, yy = np.meshgrid(np.linspace(-1, 1, ny), np.linspace(-1, 1, nx))

        self.disk = xx**2 + yy**2
        self.mask0 = self.disk < 1.0
        im_fft0 = _decorr_fft(self.image)
        norm_factor = np.abs(im_fft0)
        im_fft0_normalized = np.divide(im_fft0, norm_factor, out=np.zeros_like(im_fft0, dtype=np.complex128), where=norm_factor!=0)
        im_fft0_normalized[~np.isfinite(im_fft0_normalized)] = 0
        self.im_fft0 = im_fft0_normalized * self.mask0

        img_mean = self.image.mean(); img_std = self.image.std()
        if img_std < 1e-9: img_std = 1.0
        image_bar = (self.image - img_mean) / img_std
        im_fftk = _decorr_fft(image_bar) * self.mask0
        self.im_invk = _decorr_ifft(im_fftk).real
        self.im_fftr = _decorr_masked_fft(self.im_invk, self.mask0, self.size)

        res_max_corr = self.maximize_corcoef(self.im_fftr)
        self.snr0 = res_max_corr["snr"]; self.kc0 = res_max_corr["kc"]
        self.max_width = 2 / self.kc0 if self.kc0 > 1e-6 else 20.0
        self.kc = None; self.resolution = None

    def corcoef(self, radius, im_fftr, c1=None):
        mask = self.disk < radius**2
        f_im_fft_full = (mask * self.im_fft0).ravel()
        num_elements_to_take = self.size // 2
        if len(f_im_fft_full) < num_elements_to_take : num_elements_to_take = len(f_im_fft_full)
        f_im_fft = f_im_fft_full[:num_elements_to_take]
        c1_val = np.linalg.norm(im_fftr) if c1 is None else c1
        c2_val = np.linalg.norm(f_im_fft)
        if c1_val * c2_val < 1e-9: return 0.0
        return (im_fftr * f_im_fft.conjugate()).real.sum() / (c1_val * c2_val)

    def maximize_corcoef(self, im_fftr, r_min=0.0, r_max=1.0):
        def anti_cor(radius):
            c1_norm = np.linalg.norm(im_fftr)
            if c1_norm < 1e-9: return 1.0
            return 1 - self.corcoef(radius, im_fftr, c1=c1_norm)

        r_min_actual = max(1e-3, r_min); r_max_actual = min(1.0-1e-3, r_max)
        if r_min_actual >= r_max_actual:
             return {"snr": 0.0, "kc": 0.0}

        res = minimize_scalar(anti_cor, bounds=(r_min_actual, r_max_actual), method="bounded", options={"xatol": 1e-4})
        if not res.success or res.fun is None: return {"snr": 0.0, "kc": 0.0}
        final_snr = 1 - res.fun; final_kc = res.x
        if final_snr < 1e-3: final_kc = 0.0
        return {"snr": final_snr, "kc": final_kc}

    def compute_resolution(self):
        if self.snr0 < 1e-3 and self.kc0 < 1e-3:
            self.resolution = np.inf; self.kc = 0.0
            return None, {"snr": self.snr0, "kc": self.kc0}

        def filtered_decorr_cost(width, return_gm=True):
            f_im = self.im_invk - gaussian_filter(self.im_invk, width)
            f_im_fft = _decorr_masked_fft(f_im, self.mask0, self.size)
            res_corr = self.maximize_corcoef(f_im_fft)
            if return_gm:
                if res_corr["kc"] < 1e-6 or res_corr["snr"] < 1e-3: return 1.0
                return 1 - (res_corr["kc"] * res_corr["snr"]) ** 0.5
            return res_corr

        lower_bound, upper_bound = 0.15, self.max_width
        if upper_bound <= lower_bound:
            self.kc = self.kc0; self.resolution = 2 * self.pixel_size / self.kc if self.kc > 1e-6 else np.inf
            return None, {"snr": self.snr0, "kc": self.kc0}

        res_opt = minimize_scalar(filtered_decorr_cost, method="bounded", bounds=(lower_bound, upper_bound), options={"xatol": 1e-3})

        if not res_opt.success or res_opt.fun is None:
            self.kc = self.kc0; self.resolution = 2 * self.pixel_size / self.kc if self.kc > 1e-6 else np.inf
            return res_opt, {"snr": self.snr0, "kc": self.kc0}

        optimal_width = res_opt.x
        max_cor_at_optimal_width = filtered_decorr_cost(optimal_width, return_gm=False)
        self.kc = max_cor_at_optimal_width["kc"]
        self.resolution = 2 * self.pixel_size / self.kc if self.kc > 1e-6 else np.inf
        return res_opt, max_cor_at_optimal_width

def decorr_measure_single_image(image, pixel_size_units, square_crop=True):
    if image.ndim != 2: raise ValueError("Image must be 2D for decorr_measure_single_image.")
    imdecor = DecorrImageDecorr(image, pixel_size=pixel_size_units, square_crop=square_crop)
    _, _ = imdecor.compute_resolution()
    return {"SNR": imdecor.snr0, "resolution": imdecor.resolution}

def _decorr_analyze_single_projection(
    projection_image, projection_name_full,
    pixel_size_y_display, pixel_size_x_display, pixel_size_for_resolution_scaling,
    display_axis_labels, resolution_description, units_label="units",
    show_decorr_plots=True, log_queue=None, plot_output_dir=None
):
    num_rows, num_cols = projection_image.shape
    x_max_phys = num_cols * pixel_size_x_display
    y_max_phys = num_rows * pixel_size_y_display

    if show_decorr_plots and plot_output_dir:
        try:
            fig, ax = plt.subplots(figsize=(7, 7 * (y_max_phys / x_max_phys) if x_max_phys > 0 else 7))
            im = ax.imshow(projection_image, cmap='gray', extent=[0, x_max_phys, y_max_phys, 0])
            ax.set_title(projection_name_full); ax.set_xlabel(f"Original {display_axis_labels[1]} ({units_label})")
            ax.set_ylabel(f"Original {display_axis_labels[0]} ({units_label})"); ax.set_aspect('equal', adjustable='box')
            plt.colorbar(im, ax=ax, label="Intensity", shrink=0.8); plt.tight_layout()

            safe_proj_name = "".join(c if c.isalnum() else "_" for c in projection_name_full)
            plot_filename = Path(plot_output_dir) / f"decorr_mip_{safe_proj_name}.png"
            plt.savefig(plot_filename)
            log_message_gui(f"Saved decorrelation MIP plot to {plot_filename}", log_queue)
            plt.close(fig)
        except Exception as e:
            log_message_gui(f"Error generating/saving decorr MIP plot for {projection_name_full}: {e}", log_queue, "ERROR")


    log_message_gui(f"Calculating Decorrelation Resolution/SNR for '{projection_name_full}'...", log_queue)
    analysis_output = None
    try:
        if np.all(projection_image == projection_image[0,0]):
             log_message_gui(f"Warning (Decorrelation): Image '{projection_name_full}' is flat.", log_queue, "WARNING")

        raw_results = decorr_measure_single_image(projection_image, pixel_size_for_resolution_scaling)
        resolution_val = raw_results.get('resolution'); snr_val = raw_results.get('SNR')

        log_message_gui(f"  Calculated Decorr Resolution ({resolution_description}): "
              f"{resolution_val:.2f} ({units_label})" if resolution_val is not None and np.isfinite(resolution_val) else f"N/A or {resolution_val}", log_queue)
        log_message_gui(f"  Calculated Decorr SNR: {snr_val:.2f}" if snr_val is not None else "N/A", log_queue)
        analysis_output = {"resolution": resolution_val, "SNR": snr_val}
    except Exception as e:
        log_message_gui(f"Error during decorrelation calculation for '{projection_name_full}': {e}", log_queue, "ERROR")
        analysis_output = {"resolution": None, "SNR": None, "error": str(e)}
    return analysis_output

def run_decorrelation_analysis(
    deskewed_tiff_path, stack_name_prefix,
    lateral_pixel_size_units, axial_pixel_size_units, units_label="units",
    show_decorr_plots=True, log_queue=None, main_output_folder=None
):
    plot_output_dir_decorr = None
    if main_output_folder and show_decorr_plots:
        plot_output_dir_decorr = Path(main_output_folder) / "analysis_plots"
        plot_output_dir_decorr.mkdir(parents=True, exist_ok=True)

    try:
        image_stack = tifffile.imread(deskewed_tiff_path)
        log_message_gui(f"\nLoaded deskewed stack for decorrelation: '{deskewed_tiff_path}', shape: {image_stack.shape}", log_queue)
    except Exception as e:
        log_message_gui(f"Error loading deskewed TIFF for decorrelation: {e}", log_queue, "ERROR"); return None

    if image_stack.ndim != 3:
        log_message_gui(f"Error (Decorrelation): Deskewed stack is not 3D (shape: {image_stack.shape}). Cannot perform MIP analysis.", log_queue, "ERROR"); return None

    nz, ny, nx = image_stack.shape
    all_results = {}

    log_message_gui(f"\n--- Analyzing Z-MIP (XY Plane) with Decorrelation ---", log_queue)
    proj_xy = np.max(image_stack, axis=0)
    all_results["Z-MIP (XY Plane)"] = _decorr_analyze_single_projection(
        projection_image=proj_xy, projection_name_full=f"Z-MIP (XY) of {stack_name_prefix}",
        pixel_size_y_display=lateral_pixel_size_units, pixel_size_x_display=lateral_pixel_size_units,
        pixel_size_for_resolution_scaling=lateral_pixel_size_units,
        display_axis_labels=("Y", "X"), resolution_description="Lateral (XY)",
        units_label=units_label, show_decorr_plots=show_decorr_plots, log_queue=log_queue,
        plot_output_dir=plot_output_dir_decorr
    )

    log_message_gui(f"\n--- Analyzing Y-MIP (XZ Plane) with Decorrelation ---", log_queue)
    proj_xz = np.max(image_stack, axis=1)
    all_results["Y-MIP (XZ Plane)"] = _decorr_analyze_single_projection(
        projection_image=proj_xz, projection_name_full=f"Y-MIP (XZ) of {stack_name_prefix}",
        pixel_size_y_display=axial_pixel_size_units, pixel_size_x_display=lateral_pixel_size_units,
        pixel_size_for_resolution_scaling=axial_pixel_size_units,
        display_axis_labels=("Z", "X"), resolution_description="Axial-like (Z in XZ)",
        units_label=units_label, show_decorr_plots=show_decorr_plots, log_queue=log_queue,
        plot_output_dir=plot_output_dir_decorr
    )

    log_message_gui(f"\n--- Analyzing X-MIP (YZ Plane) with Decorrelation ---", log_queue)
    proj_yz = np.max(image_stack, axis=2)
    all_results["X-MIP (YZ Plane)"] = _decorr_analyze_single_projection(
        projection_image=proj_yz, projection_name_full=f"X-MIP (YZ) of {stack_name_prefix}",
        pixel_size_y_display=axial_pixel_size_units, pixel_size_x_display=lateral_pixel_size_units,
        pixel_size_for_resolution_scaling=axial_pixel_size_units,
        display_axis_labels=("Z", "Y"), resolution_description="Axial-like (Z in YZ)",
        units_label=units_label, show_decorr_plots=show_decorr_plots, log_queue=log_queue,
        plot_output_dir=plot_output_dir_decorr
    )

    del image_stack, proj_xy, proj_xz, proj_yz; gc.collect()
    return all_results




# --- GPU Memory Management ---
def clear_gpu_memory():
    try:
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
    except Exception as e:
        print(f"Minor error during clear_gpu_memory: {e}")

# --- Logging Function ---
def log_message_gui(message, log_queue=None, level="INFO"):
    formatted_message = f"[{time.strftime('%H:%M:%S')}] [{level}] {message}"
    if log_queue: log_queue.put(formatted_message + "\n")
    else: print(formatted_message)

# --- PSF Helper Functions ---
def psf_gaussian(x, amplitude, center, sigma_related_width):
    return amplitude * np.exp(-((x - center) / sigma_related_width) ** 2)

def psf_calculate_fwhm_from_fit(popt, pixel_size_val):
    return popt[2] * 2.0 * np.sqrt(np.log(2.0)) * pixel_size_val

# --- Main PSF Fitting Analysis Function ---
def run_psf_fitting_analysis(
    deskewed_tiff_path, base_file_name,
    pixel_size_z_nm, pixel_size_y_nm, pixel_size_x_nm,
    padding_pixels=1, roi_radius_pixels=15, intensity_threshold=1000.0,
    fit_quality_r2_threshold=0.85, show_psf_plots=True, show_psf_threshold_plot=True,
    log_queue=None, main_output_folder=None, psf_plot_path_queue=None,
    use_gpu_for_psf_prep=False 
):
    log_message_gui(f"\n--- Starting PSF Fitting Analysis for: {base_file_name} (GPU Prep: {use_gpu_for_psf_prep}) ---", log_queue)

    plot_output_dir_psf = None
    if main_output_folder and (show_psf_plots or show_psf_threshold_plot):
        plot_output_dir_psf = Path(main_output_folder) / "analysis_plots"
        plot_output_dir_psf.mkdir(parents=True, exist_ok=True)

    final_fwhm_plot_path = None
    
    # --- Initialize variables that might be accessed in `finally` ---
    im_data_raw_cpu = None
    im_data = None # This will hold the padded data, either CPU or GPU
    labeled_image = None # This will hold the labeled image, either CPU or GPU
    im_bw = None # This will hold the thresholded image, either CPU or GPU
    # GPU specific intermediate arrays for cleanup
    im_data_gpu_intermediate = None 
    im_bw_gpu_intermediate = None # If im_bw becomes a distinct GPU array

    xp = np # Default to numpy
    label_func = scipy_label # Default to scipy label
    gpu_prep_active = False # Flag to track if GPU path was successfully used for prep

    psf_results = None # Initialize to ensure it's always returned

    try:
        im_data_raw_cpu = tifffile.imread(deskewed_tiff_path).astype(np.float64)
        log_message_gui(f"Loaded deskewed stack for PSF: '{deskewed_tiff_path}', shape: {im_data_raw_cpu.shape}", log_queue)

        if im_data_raw_cpu.ndim != 3:
            log_message_gui(f"Error (PSF): Image not 3D. Dim: {im_data_raw_cpu.ndim}. Skipping.", log_queue, "ERROR")
            if psf_plot_path_queue: psf_plot_path_queue.put(None); return None # Ensure queue gets None
            return None
        if im_data_raw_cpu.size == 0:
            log_message_gui(f"Error (PSF): Image is empty. Skipping.", log_queue, "ERROR")
            if psf_plot_path_queue: psf_plot_path_queue.put(None); return None
            return None

        if use_gpu_for_psf_prep:
            try:
                cp.cuda.Device(0).use() 
                xp = cp
                label_func = ndi_cp.label 
                im_data_gpu_intermediate = xp.asarray(im_data_raw_cpu)
                pad_width_gpu = [(padding_pixels, padding_pixels)] * im_data_gpu_intermediate.ndim
                im_data = xp.pad(im_data_gpu_intermediate, pad_width_gpu, mode="constant", constant_values=0)
                # im_data_gpu_intermediate is now aliased or copied into im_data (if on GPU)
                # No need to delete im_data_gpu_intermediate yet if im_data is just a view or same object.
                # If xp.pad always copies, then im_data_gpu_intermediate could be deleted.
                # For safety, we'll manage it in finally.
                gpu_prep_active = True
                log_message_gui(f"PSF data transferred to GPU and padded. Shape: {im_data.shape}", log_queue)
            except Exception as e_gpu_init:
                log_message_gui(f"PSF GPU Prep FAILED: {type(e_gpu_init).__name__} - {e_gpu_init}. Falling back to CPU.", log_queue, "WARNING")
                xp = np; label_func = scipy_label; gpu_prep_active = False
                if im_data_gpu_intermediate is not None: del im_data_gpu_intermediate; im_data_gpu_intermediate = None # Clean up if partly made
                clear_gpu_memory() # Clear if GPU error
        
        if not gpu_prep_active: 
            xp = np; label_func = scipy_label
            pad_width_cpu = [(padding_pixels, padding_pixels)] * im_data_raw_cpu.ndim
            im_data = xp.pad(im_data_raw_cpu, pad_width_cpu, mode="constant", constant_values=0)
            log_message_gui(f"PSF data padded on CPU. Shape: {im_data.shape}", log_queue)

        # im_data_raw_cpu is used to create im_data (either GPU or CPU version)
        # We can delete im_data_raw_cpu now as its content is in im_data or im_data_gpu_intermediate
        if im_data_raw_cpu is not None: del im_data_raw_cpu; im_data_raw_cpu = None; gc.collect()

        im_bw = im_data > intensity_threshold
        if gpu_prep_active: im_bw_gpu_intermediate = im_bw # Keep ref for cleanup if GPU

        if show_psf_threshold_plot and plot_output_dir_psf and im_data.shape[0] > 1 :
            try:
                im_bw_cpu_for_plot = cp.asnumpy(im_bw) if gpu_prep_active else im_bw.copy() # Ensure CPU copy
                fig_thresh, ax_thresh = plt.subplots(figsize=(6,6))
                ax_thresh.imshow(np.max(im_bw_cpu_for_plot, axis=0), cmap="gray")
                ax_thresh.set_title(f"PSF Thresholded Max Projection (Z): {base_file_name}")
                safe_base_name = "".join(c if c.isalnum() else"_" for c in base_file_name)
                plot_filename = plot_output_dir_psf / f"psf_threshold_proj_{safe_base_name}.png"
                plt.savefig(plot_filename); plt.close(fig_thresh)
                log_message_gui(f"Saved PSF threshold plot to {plot_filename}", log_queue)
                del im_bw_cpu_for_plot 
            except Exception as e: log_message_gui(f"Error PSF threshold plot: {e}", log_queue, "ERROR")
        
        labeled_image, num_labels = label_func(im_bw) # `label_func` is cp or scipy
        if num_labels == 0:
            log_message_gui("PSF: No particles after thresholding.", log_queue, "INFO")
            if psf_plot_path_queue: psf_plot_path_queue.put(None); return None
            return None
        log_message_gui(f"PSF: {num_labels} potential particles by label func.", log_queue)

        psf_results_lists = {k: [] for k in ["fwhmZ_nm","fwhmY_nm","fwhmX_nm","zR2","yR2","xR2", 
                                           "peak_intensity","centroid_z_px","centroid_y_px","centroid_x_px"]}
        im_roi_gpu = None # For finally block within loop

        if gpu_prep_active:
            log_message_gui("PSF: Iterating labels for GPU-based ROI extraction.", log_queue, "DEBUG")
            for label_idx in range(1, num_labels + 1):
                try:
                    current_label_mask_gpu = (labeled_image == label_idx)
                    if not xp.any(current_label_mask_gpu): continue
                    coords_gpu = xp.argwhere(current_label_mask_gpu)
                    centroid_gpu_int = xp.round(xp.mean(coords_gpu.astype(xp.float32),axis=0)).astype(xp.int32)
                    del coords_gpu, current_label_mask_gpu # Free GPU mem
                    
                    centroid_px_zyx_cpu = cp.asnumpy(centroid_gpu_int)
                    del centroid_gpu_int

                    start_idx = centroid_px_zyx_cpu - roi_radius_pixels
                    end_idx = centroid_px_zyx_cpu + roi_radius_pixels + 1
                    
                    im_data_shape_cpu = im_data.shape # Get shape for bounds check
                    if np.any(start_idx < 0) or np.any(end_idx > np.array(im_data_shape_cpu)): continue
                    
                    slicer_gpu=(slice(start_idx[0],end_idx[0]),slice(start_idx[1],end_idx[1]),slice(start_idx[2],end_idx[2]))
                    im_roi_gpu = im_data[slicer_gpu]
                    if im_roi_gpu.size == 0 or xp.all(im_roi_gpu == 0):
                        if im_roi_gpu is not None: del im_roi_gpu; im_roi_gpu = None; continue
                    
                    max_idx_gpu_flat = xp.argmax(im_roi_gpu)
                    max_intensity_in_roi_idx_zyx_gpu = xp.unravel_index(max_idx_gpu_flat, im_roi_gpu.shape)
                    peak_val_gpu = im_roi_gpu[max_intensity_in_roi_idx_zyx_gpu]
                    peak_val_cpu = float(cp.asnumpy(peak_val_gpu))
                    # --- CORRECTED TRANSFER FOR TUPLE OF ARRAYS ---
                    # --- CORRECTED TUPLE CONVERSION ---
                    max_idx_cpu_list = []
                    for idx_arr_gpu in max_intensity_in_roi_idx_zyx_gpu:
                        # idx_arr_gpu is a 0-D CuPy array, e.g., cp.array(5)
                        max_idx_cpu_list.append(idx_arr_gpu.item()) # .item() converts 0-D array to Python scalar
                    max_idx_cpu = tuple(max_idx_cpu_list)
                    # Now max_idx_cpu is a tuple of Python ints, e.g., (5, 10, 12)

                    if not (0<=max_idx_cpu[0]<im_roi_gpu.shape[0] and \
                            0<=max_idx_cpu[1]<im_roi_gpu.shape[1] and \
                            0<=max_idx_cpu[2]<im_roi_gpu.shape[2]): continue

                    line_z_prof_cpu = cp.asnumpy(im_roi_gpu[:,max_idx_cpu[1],max_idx_cpu[2]])
                    line_y_prof_cpu = cp.asnumpy(im_roi_gpu[max_idx_cpu[0],:,max_idx_cpu[2]])
                    line_x_prof_cpu = cp.asnumpy(im_roi_gpu[max_idx_cpu[0],max_idx_cpu[1],:])
                
                finally: # Ensure im_roi_gpu is deleted for this iteration
                    if im_roi_gpu is not None: del im_roi_gpu; im_roi_gpu = None
                
                # --- Fitting remains on CPU (using _cpu arrays) ---
                def normalize_profile(line):min_v,max_v=np.min(line),np.max(line);return(line-min_v)/(max_v-min_v)if(max_v-min_v)>1e-9 else np.zeros_like(line)
                line_z_norm=normalize_profile(line_z_prof_cpu);line_y_norm=normalize_profile(line_y_prof_cpu);line_x_norm=normalize_profile(line_x_prof_cpu)
                fwhm_z,fwhm_y,fwhm_x=np.nan,np.nan,np.nan;r2_z,r2_y,r2_x=np.nan,np.nan,np.nan
                coords_z,coords_y,coords_x=np.arange(line_z_norm.size),np.arange(line_y_norm.size),np.arange(line_x_norm.size)
                max_fit_w_px=roi_radius_pixels+5.0
                def fit_line_psf(coords,line_norm,px_size_nm):
                    fwhm,r2=np.nan,np.nan
                    if len(line_norm)<4:return fwhm,r2
                    peak_idx=float(np.argmax(line_norm))
                    if not(0.25*len(coords)<peak_idx<0.75*len(coords)):peak_idx=float(len(coords)//2)
                    p0=[1.0,peak_idx,1.7];bounds=([0.0,0.0,0.1],[1.5,float(len(coords)-1),max_fit_w_px])
                    try:
                        popt,pcov=curve_fit(psf_gaussian,coords,line_norm,p0=p0,bounds=bounds,maxfev=5000)
                        fit_line=psf_gaussian(coords,*popt);ss_res=np.sum((line_norm-fit_line)**2);ss_tot=np.sum((line_norm-np.mean(line_norm))**2)
                        r2=1-(ss_res/ss_tot)if ss_tot>1e-9 else(1.0 if ss_res<1e-9 else 0.0)
                        if r2>-np.inf:fwhm=psf_calculate_fwhm_from_fit(popt,px_size_nm)
                    except RuntimeError:pass
                    except ValueError:pass
                    return fwhm,r2
                if im_data.shape[0]>1:fwhm_z,r2_z=fit_line_psf(coords_z,line_z_norm,pixel_size_z_nm)
                fwhm_y,r2_y=fit_line_psf(coords_y,line_y_norm,pixel_size_y_nm)
                fwhm_x,r2_x=fit_line_psf(coords_x,line_x_norm,pixel_size_x_nm)
                if not(np.isnan(r2_x)and np.isnan(r2_y)and np.isnan(r2_z)):
                    psf_results_lists["fwhmZ_nm"].append(fwhm_z);psf_results_lists["fwhmY_nm"].append(fwhm_y);psf_results_lists["fwhmX_nm"].append(fwhm_x)
                    psf_results_lists["zR2"].append(r2_z);psf_results_lists["yR2"].append(r2_y);psf_results_lists["xR2"].append(r2_x)
                    psf_results_lists["peak_intensity"].append(peak_val_cpu)
                    psf_results_lists["centroid_z_px"].append(centroid_px_zyx_cpu[0])
                    psf_results_lists["centroid_y_px"].append(centroid_px_zyx_cpu[1])
                    psf_results_lists["centroid_x_px"].append(centroid_px_zyx_cpu[2])
        else: # CPU path for regionprops
            # This is the original SciPy/Scikit-image regionprops loop
            # Ensure im_data and labeled_image are NumPy arrays for this path
            if isinstance(im_data, cp.ndarray): im_data = cp.asnumpy(im_data) # Should not happen if gpu_prep_active is False
            if isinstance(labeled_image, cp.ndarray): labeled_image = cp.asnumpy(labeled_image)

            log_message_gui("PSF: Using SciPy regionprops for particle analysis.", log_queue, "DEBUG")
            # Import locally if not already global to avoid conflict if cucim is used elsewhere
            from skimage.measure import regionprops as skimage_regionprops 
            stats_cpu = skimage_regionprops(labeled_image, intensity_image=im_data)
            log_message_gui(f"PSF: {len(stats_cpu)} particles from regionprops.", log_queue)
            for particle in stats_cpu: # Iterate through CPU regionprops
                # ... (The exact same CPU-based fitting loop as in your original code)
                centroid_px_zyx_cpu = np.round(particle.centroid).astype(int)
                start_idx = centroid_px_zyx_cpu - roi_radius_pixels
                end_idx = centroid_px_zyx_cpu + roi_radius_pixels + 1
                if np.any(start_idx < 0) or np.any(end_idx > np.array(im_data.shape)): continue
                im_roi_cpu = im_data[start_idx[0]:end_idx[0],start_idx[1]:end_idx[1],start_idx[2]:end_idx[2]]
                if im_roi_cpu.size == 0 or np.all(im_roi_cpu == 0): continue
                try: max_idx_cpu = np.unravel_index(np.argmax(im_roi_cpu), im_roi_cpu.shape)
                except ValueError: continue
                peak_val_cpu = im_roi_cpu[max_idx_cpu]
                line_z_prof_cpu=im_roi_cpu[:,max_idx_cpu[1],max_idx_cpu[2]]
                line_y_prof_cpu=im_roi_cpu[max_idx_cpu[0],:,max_idx_cpu[2]]
                line_x_prof_cpu=im_roi_cpu[max_idx_cpu[0],max_idx_cpu[1],:]
                def normalize_profile(line):min_v,max_v=np.min(line),np.max(line);return(line-min_v)/(max_v-min_v)if(max_v-min_v)>1e-9 else np.zeros_like(line)
                line_z_norm=normalize_profile(line_z_prof_cpu);line_y_norm=normalize_profile(line_y_prof_cpu);line_x_norm=normalize_profile(line_x_prof_cpu)
                fwhm_z,fwhm_y,fwhm_x=np.nan,np.nan,np.nan;r2_z,r2_y,r2_x=np.nan,np.nan,np.nan
                coords_z,coords_y,coords_x=np.arange(line_z_norm.size),np.arange(line_y_norm.size),np.arange(line_x_norm.size)
                max_fit_w_px=roi_radius_pixels+5.0
                def fit_line_psf(coords,line_norm,px_size_nm):
                    fwhm,r2=np.nan,np.nan
                    if len(line_norm)<4:return fwhm,r2
                    peak_idx=float(np.argmax(line_norm))
                    if not(0.25*len(coords)<peak_idx<0.75*len(coords)):peak_idx=float(len(coords)//2)
                    p0=[1.0,peak_idx,1.7];bounds=([0.0,0.0,0.1],[1.5,float(len(coords)-1),max_fit_w_px])
                    try:
                        popt,pcov=curve_fit(psf_gaussian,coords,line_norm,p0=p0,bounds=bounds,maxfev=5000)
                        fit_line=psf_gaussian(coords,*popt);ss_res=np.sum((line_norm-fit_line)**2);ss_tot=np.sum((line_norm-np.mean(line_norm))**2)
                        r2=1-(ss_res/ss_tot)if ss_tot>1e-9 else(1.0 if ss_res<1e-9 else 0.0)
                        if r2>-np.inf:fwhm=psf_calculate_fwhm_from_fit(popt,px_size_nm)
                    except RuntimeError:pass
                    except ValueError:pass
                    return fwhm,r2
                if im_data.shape[0]>1:fwhm_z,r2_z=fit_line_psf(coords_z,line_z_norm,pixel_size_z_nm)
                fwhm_y,r2_y=fit_line_psf(coords_y,line_y_norm,pixel_size_y_nm)
                fwhm_x,r2_x=fit_line_psf(coords_x,line_x_norm,pixel_size_x_nm)
                if not(np.isnan(r2_x)and np.isnan(r2_y)and np.isnan(r2_z)):
                    psf_results_lists["fwhmZ_nm"].append(fwhm_z);psf_results_lists["fwhmY_nm"].append(fwhm_y);psf_results_lists["fwhmX_nm"].append(fwhm_x)
                    psf_results_lists["zR2"].append(r2_z);psf_results_lists["yR2"].append(r2_y);psf_results_lists["xR2"].append(r2_x)
                    psf_results_lists["peak_intensity"].append(peak_val_cpu)
                    psf_results_lists["centroid_z_px"].append(centroid_px_zyx_cpu[0])
                    psf_results_lists["centroid_y_px"].append(centroid_px_zyx_cpu[1])
                    psf_results_lists["centroid_x_px"].append(centroid_px_zyx_cpu[2])

        # Convert lists to NumPy arrays (CPU)
        psf_results = {"file_name": base_file_name}
        for key, lst_data in psf_results_lists.items():
            psf_results[key] = np.array(lst_data if lst_data else [], dtype=np.float64) # Ensure float for nanmean/nanstd

        # --- Filtering and summary stats (remains on CPU) ---
        # ... (Your existing robust filtering logic based on R2 and has_any_fwhm)
        # ... (Calculation of mean/std FWHM)
        # ... (Plotting FWHM vs Pos)
        # This part of the code should be largely the same as your previous working version.
        # For brevity, I am omitting the direct copy-paste of the extensive filtering and plotting here.
        # Ensure it uses `psf_results["xR2"]`, `psf_results["fwhmX_nm"]`, etc.
        
        # Example placeholder for the filtering and summary section:
        if psf_results["fwhmX_nm"].size > 0: # Check if any beads were processed
            q_x = np.where(~np.isnan(psf_results["xR2"]), psf_results["xR2"] > fit_quality_r2_threshold, True)
            q_y = np.where(~np.isnan(psf_results["yR2"]), psf_results["yR2"] > fit_quality_r2_threshold, True)
            q_z = np.where(~np.isnan(psf_results["zR2"]), psf_results["zR2"] > fit_quality_r2_threshold, True)
            has_fwhm = (~np.isnan(psf_results["fwhmX_nm"]))| (~np.isnan(psf_results["fwhmY_nm"]))| (~np.isnan(psf_results["fwhmZ_nm"]))
            good_indices = np.where(has_fwhm & q_x & q_y & q_z)[0]
            num_good_beads = len(good_indices)
            log_message_gui(f"PSF: {psf_results['fwhmX_nm'].size} beads processed, {num_good_beads} good beads after R2 filter.", log_queue)

            psf_results["filtered_fwhmX_nm"] = psf_results["fwhmX_nm"][good_indices]
            # ... (populate other filtered_... arrays) ...
            psf_results["filtered_fwhmY_nm"] = psf_results["fwhmY_nm"][good_indices]
            psf_results["filtered_fwhmZ_nm"] = psf_results["fwhmZ_nm"][good_indices]
            psf_results["filtered_peak_intensity"] = psf_results["peak_intensity"][good_indices]
            psf_results["filtered_centroid_x_um"] = psf_results["centroid_x_px"][good_indices] * pixel_size_x_nm / 1000.0
            psf_results["filtered_centroid_y_um"] = psf_results["centroid_y_px"][good_indices] * pixel_size_y_nm / 1000.0
            psf_results["filtered_centroid_z_um"] = psf_results["centroid_z_px"][good_indices] * pixel_size_z_nm / 1000.0


            mean_x = np.nanmean(psf_results["filtered_fwhmX_nm"]) if num_good_beads > 0 else np.nan
            # ... (calculate other means and stds)
            std_x = np.nanstd(psf_results["filtered_fwhmX_nm"]) if num_good_beads > 0 else np.nan
            mean_y = np.nanmean(psf_results["filtered_fwhmY_nm"]) if num_good_beads > 0 else np.nan
            std_y = np.nanstd(psf_results["filtered_fwhmY_nm"]) if num_good_beads > 0 else np.nan
            mean_z = np.nanmean(psf_results["filtered_fwhmZ_nm"]) if num_good_beads > 0 else np.nan
            std_z = np.nanstd(psf_results["filtered_fwhmZ_nm"]) if num_good_beads > 0 else np.nan

            log_message_gui(f"PSF Summary: X:{mean_x:.1f}±{std_x:.1f}nm, Y:{mean_y:.1f}±{std_y:.1f}nm, Z:{mean_z:.1f}±{std_z:.1f}nm ({num_good_beads} beads)", log_queue)
            psf_results["summary_stats"] = {"mean_fwhm_x_nm":mean_x,"std_fwhm_x_nm":std_x, #... etc.
                                            "mean_fwhm_y_nm":mean_y,"std_fwhm_y_nm":std_y,
                                            "mean_fwhm_z_nm":mean_z,"std_fwhm_z_nm":std_z,
                                            "num_good_beads":num_good_beads}
            
            if show_psf_plots and plot_output_dir_psf and num_good_beads > 0:
                fig_fwhm = None # Initialize for finally block in case of error during plot creation
                try:
                    fig_fwhm, axs_fwhm = plt.subplots(3, 1, figsize=(8, 12), sharex=True) # Increased height slightly
                    
                    # Ensure filtered_peak_intensity has the same length as other filtered arrays if used for color
                    # This should be guaranteed if good_indices was applied consistently.
                    # For safety, we'll use the length of the shortest valid array for scatter plotting.
                    
                    plot_data_map = {
                        "X": {"fwhm": psf_results["filtered_fwhmX_nm"], 
                              "pos": psf_results["filtered_centroid_x_um"], 
                              "label": "X FWHM vs X-pos"},
                        "Y": {"fwhm": psf_results["filtered_fwhmY_nm"], 
                              "pos": psf_results["filtered_centroid_x_um"], # Assuming all vs X-pos
                              "label": "Y FWHM vs X-pos"},
                        "Z": {"fwhm": psf_results["filtered_fwhmZ_nm"], 
                              "pos": psf_results["filtered_centroid_x_um"], # Assuming all vs X-pos
                              "label": "Z FWHM vs X-pos"}
                    }
                    
                    # Common peak intensity data for coloring, ensure it matches the filtered beads
                    peak_intensities_filtered = psf_results.get("filtered_peak_intensity", np.array([]))

                    any_plot_has_data = False
                    for i, (axis_key, data_dict) in enumerate(plot_data_map.items()):
                        ax = axs_fwhm[i]
                        fwhm_data = data_dict["fwhm"]
                        pos_data = data_dict["pos"]
                        plot_title = data_dict["label"]

                        if fwhm_data.size > 0 and pos_data.size > 0 and fwhm_data.size == pos_data.size:
                            # Further ensure peak_intensities_filtered aligns if used for color
                            # We'll take the intersection of valid data for all three arrays
                            
                            # Create a mask for valid (non-NaN) data points across all arrays needed for this subplot
                            valid_mask = ~np.isnan(fwhm_data) & ~np.isnan(pos_data)
                            if peak_intensities_filtered.size == fwhm_data.size: # Only use intensity if lengths match
                                valid_mask &= ~np.isnan(peak_intensities_filtered)
                            
                            valid_fwhm = fwhm_data[valid_mask]
                            valid_pos = pos_data[valid_mask]
                            
                            if valid_fwhm.size > 0: # If there's anything left to plot
                                any_plot_has_data = True
                                if peak_intensities_filtered.size == fwhm_data.size:
                                    valid_intensity = peak_intensities_filtered[valid_mask]
                                    vmin_calc = np.percentile(valid_intensity, 5) if valid_intensity.size > 0 else None
                                    vmax_calc = np.percentile(valid_intensity, 95) if valid_intensity.size > 0 else None
                                    sc = ax.scatter(valid_pos, valid_fwhm, c=valid_intensity, 
                                                    cmap='viridis', alpha=0.7, edgecolors='k', linewidths=0.5,
                                                    vmin=vmin_calc, vmax=vmax_calc)
                                    # Add colorbar to the figure, associated with this axes object
                                    # To avoid multiple colorbars, one could be added to the figure globally,
                                    # or we can add one per subplot if scales differ significantly.
                                    # For now, let's add one per subplot if data allows.
                                    cbar = fig_fwhm.colorbar(sc, ax=ax, label='Peak Intensity', aspect=15, pad=0.03, fraction=0.046)
                                    cbar.ax.tick_params(labelsize=8)
                                else: # Plot without color if intensity data is mismatched or missing
                                    ax.scatter(valid_pos, valid_fwhm, alpha=0.7, edgecolors='k', linewidths=0.5, c='blue')

                                y_upper_limit = np.nanmax(valid_fwhm) * 1.2 if np.any(np.isfinite(valid_fwhm)) and np.nanmax(valid_fwhm) > 0 else 100
                                ax.set_ylim(bottom=0, top=y_upper_limit)
                            else:
                                ax.text(0.5, 0.5, 'No valid data points', ha='center', va='center', transform=ax.transAxes)
                        else:
                            ax.text(0.5, 0.5, 'Insufficient data for plot', ha='center', va='center', transform=ax.transAxes)
                        
                        ax.set_title(f"{plot_title} - {base_file_name}", fontsize=10)
                        ax.set_xlabel("X Position (µm)" if i == 2 else "") # Only show x-label on bottom plot due to sharex
                        ax.set_ylabel("FWHM (nm)")
                        ax.grid(True, linestyle=':', alpha=0.7)
                    
                    if not any_plot_has_data:
                        log_message_gui("PSF Plot: No valid data points found for any FWHM subplot.", log_queue, "WARNING")
                        plt.close(fig_fwhm) # Close the empty figure
                        fig_fwhm = None # Mark as None so it's not saved
                    else:
                        fig_fwhm.suptitle(f"PSF FWHM Analysis: {base_file_name}", fontsize=12, y=1.02) # y for spacing
                        fig_fwhm.tight_layout(rect=[0, 0, 1, 0.98]) # rect to make space for suptitle

                    if fig_fwhm: # Only save if figure was created and has data
                        safe_base_name = "".join(c if c.isalnum() else "_" for c in base_file_name)
                        plot_filename = plot_output_dir_psf / f"psf_fwhm_vs_pos_{safe_base_name}.png"
                        plt.savefig(plot_filename, dpi=150)
                        log_message_gui(f"Saved PSF FWHM plot to {plot_filename}", log_queue)
                        final_fwhm_plot_path = str(plot_filename)
                        plt.close(fig_fwhm)

                except Exception as e_plot:
                    log_message_gui(f"Error during PSF FWHM plot generation: {type(e_plot).__name__} - {e_plot}", log_queue, "ERROR")
                    log_message_gui(traceback.format_exc(), log_queue, "DEBUG")
                    if fig_fwhm is not None and plt.fignum_exists(fig_fwhm.number): # Check if figure exists before closing
                        plt.close(fig_fwhm)
            elif num_good_beads == 0:
                 log_message_gui("PSF Plot: Skipped, no good beads found.", log_queue, "INFO")
            # Implicit else: show_psf_plots is False or plot_output_dir_psf is None

        else: # No beads processed or no good beads for summary from the start
            log_message_gui("PSF: No beads processed or no good beads for summary statistics or plotting.", log_queue, "INFO")
            psf_results["summary_stats"] = {"num_good_beads":0} # Ensure summary_stats exists

        if psf_plot_path_queue: psf_plot_path_queue.put(final_fwhm_plot_path)
   
    
    except Exception as e_psf_pipeline:
        log_message_gui(f"CRITICAL Error in PSF Fitting Pipeline: {type(e_psf_pipeline).__name__} - {e_psf_pipeline}", log_queue, "CRITICAL")
        log_message_gui(traceback.format_exc(), log_queue, "CRITICAL")
        if psf_plot_path_queue: psf_plot_path_queue.put(None) # Ensure queue gets a response
        psf_results = {"error": str(e_psf_pipeline)} # Populate with error

    finally:
        # Cleanup GPU arrays if they were used and are CuPy arrays
        # Check actual instance type before del if var can be np or cp
        if im_data_gpu_intermediate is not None: del im_data_gpu_intermediate; im_data_gpu_intermediate = None
        if im_bw_gpu_intermediate is not None: del im_bw_gpu_intermediate; im_bw_gpu_intermediate = None
        
        if 'im_data' in locals() and im_data is not None and isinstance(im_data, cp.ndarray): del im_data; im_data = None
        elif 'im_data' in locals() and im_data is not None: del im_data; im_data = None # CPU case

        if 'labeled_image' in locals() and labeled_image is not None and isinstance(labeled_image, cp.ndarray): del labeled_image; labeled_image = None
        elif 'labeled_image' in locals() and labeled_image is not None: del labeled_image; labeled_image = None # CPU case
        
        if 'im_bw' in locals() and im_bw is not None and isinstance(im_bw, cp.ndarray) and im_bw is not im_bw_gpu_intermediate : del im_bw; im_bw = None
        elif 'im_bw' in locals() and im_bw is not None and not gpu_prep_active: del im_bw; im_bw = None # CPU case, im_bw_gpu_intermediate is None

        if 'im_data_raw_cpu' in locals() and im_data_raw_cpu is not None: del im_data_raw_cpu # Should be None already
        
        if gpu_prep_active: clear_gpu_memory()
        gc.collect()
    
    return psf_results
# --- GUI APPLICATION CLASS ---
class OPMAnalysisApp:
    def __init__(self, master):
        self.master = master
        master.title("OPM Analysis Suite (Deskew, Decorrelation, PSF)")
        master.geometry("1200x900")

        self.filepath = tk.StringVar()
        self.log_queue = queue.Queue()
        self.image_display_queue = queue.Queue()
        self.results_queue = queue.Queue()
        self.psf_plot_path_queue = queue.Queue()
        self.analysis_thread = None # Initialize thread attribute

        # Main MIP Display state
        self.original_pil_image = None
        self.original_image_data_np = None
        self.image_data_min_val = 0
        self.image_data_max_val = 255
        self.current_zoom_level = 1.0
        self.view_rect_on_original = None
        self.pan_start_pos = None
        self.pan_start_view_offset = None
        self.tk_image = None

        # PSF Plot Display state
        self.psf_plot_original_pil_image = None
        self.psf_plot_current_zoom_level = 1.0
        self.psf_plot_view_rect_on_original = None
        self.psf_plot_pan_start_pos = None
        self.psf_plot_pan_start_view_offset = None
        self.psf_tk_plot_image = None

        # Parameter Variables (Deskewing, Decorrelation, PSF)
        self.param_deskew_dx_um = tk.DoubleVar(value=0.122)
        self.param_deskew_dz_um = tk.DoubleVar(value=0.2)
        self.param_deskew_angle_deg = tk.DoubleVar(value=45.0)
        self.param_deskew_flip_direction = tk.IntVar(value=1)
        self.param_deskew_save_intermediate_shear = tk.BooleanVar(value=False)
        self.param_deskew_show_plots = tk.BooleanVar(value=True)
        # --- NEW GUI VARIABLE for Smoothing ---
        self.param_deskew_smooth_shear = tk.BooleanVar(value=False) # Default to False
        self.param_deskew_smooth_sigma = tk.DoubleVar(value=0.7) # Default sigma
        self.param_deskew_save = tk.BooleanVar(value=True)
        
        
        self.param_decorr_units_label = tk.StringVar(value="um")
        self.param_decorr_show_plots = tk.BooleanVar(value=True)
        self.param_psf_padding_pixels = tk.IntVar(value=1)
        self.param_psf_roi_radius_pixels = tk.IntVar(value=15)
        self.param_psf_intensity_threshold = tk.DoubleVar(value=1000.0)
        self.param_psf_fit_quality_r2_threshold = tk.DoubleVar(value=0.85)
        self.param_psf_show_plots = tk.BooleanVar(value=True)
        self.param_psf_show_threshold_plot = tk.BooleanVar(value=True)

        # Display Adjustment Variables (ImageJ-like: Min and Max displayed values)
        self.display_min_slider_var = tk.DoubleVar(value=0)
        self.display_max_slider_var = tk.DoubleVar(value=255)
        self.display_actual_min_max_label_var = tk.StringVar(value="Img Min/Max: N/A")

        # --- UI Layout ---
        main_pane = ttk.PanedWindow(master, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        left_pane_frame = ttk.Frame(main_pane, padding="10")
        main_pane.add(left_pane_frame, weight=1)

        file_frame = ttk.LabelFrame(left_pane_frame, text="Input File", padding="5")
        file_frame.pack(fill=tk.X, pady=(0,10))
        ttk.Button(file_frame, text="Select TIFF File", command=self.select_file).pack(side=tk.LEFT, padx=5)
        ttk.Entry(file_frame, textvariable=self.filepath, width=40, state="readonly").pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        param_notebook = ttk.Notebook(left_pane_frame)
        param_notebook.pack(fill=tk.X, expand=False, pady=5)

        deskew_tab = ttk.Frame(param_notebook, padding="10")
        param_notebook.add(deskew_tab, text="Deskewing")
        self._create_deskew_params_ui(deskew_tab)

        decorr_tab = ttk.Frame(param_notebook, padding="10")
        param_notebook.add(decorr_tab, text="Decorrelation")
        self._create_decorr_params_ui(decorr_tab)

        psf_tab = ttk.Frame(param_notebook, padding="10")
        param_notebook.add(psf_tab, text="PSF Fitting")
        self._create_psf_params_ui(psf_tab)

        adj_frame = ttk.LabelFrame(left_pane_frame, text="MIP Display Adjustments", padding="10")
        adj_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        self._create_display_adjustments_ui(adj_frame)

        self.process_button = ttk.Button(left_pane_frame, text="Run Full Analysis", command=self.run_analysis_thread_gui, style="Accent.TButton")
        self.process_button.pack(pady=10, fill=tk.X, ipady=5)
        style = ttk.Style(); style.configure("Accent.TButton", font=('Helvetica', 10, 'bold'))

        right_pane_frame = ttk.PanedWindow(main_pane, orient=tk.VERTICAL)
        main_pane.add(right_pane_frame, weight=3)

        image_display_frame = ttk.LabelFrame(right_pane_frame, text="Combined MIP Output (Scroll=Zoom, Drag=Pan)", padding="5")
        right_pane_frame.add(image_display_frame, weight=3)
        self.image_label = ttk.Label(image_display_frame, text="Combined MIP from Deskewing will appear here", relief="groove", anchor="center", justify="center")
        self.image_label.pack(fill=tk.BOTH, expand=True)
        self.image_label.bind("<MouseWheel>", self._on_mouse_wheel_gui)
        self.image_label.bind("<Button-4>", self._on_mouse_wheel_gui)
        self.image_label.bind("<Button-5>", self._on_mouse_wheel_gui)
        self.image_label.bind("<ButtonPress-1>", self._on_button_press_gui)
        self.image_label.bind("<B1-Motion>", self._on_mouse_drag_gui)
        self.image_label.bind("<ButtonRelease-1>", self._on_button_release_gui)

        output_notebook = ttk.Notebook(right_pane_frame)
        right_pane_frame.add(output_notebook, weight=2)

        log_tab = ttk.Frame(output_notebook, padding="5")
        output_notebook.add(log_tab, text="Log")
        self.log_text_area = scrolledtext.ScrolledText(log_tab, wrap=tk.WORD, width=80, height=10)
        self.log_text_area.pack(fill=tk.BOTH, expand=True)
        self.log_text_area.configure(state='disabled')

        results_tab = ttk.Frame(output_notebook, padding="5")
        output_notebook.add(results_tab, text="Summary Results")
        self.results_text_area = scrolledtext.ScrolledText(results_tab, wrap=tk.WORD, width=80, height=10)
        self.results_text_area.pack(fill=tk.BOTH, expand=True)
        self.results_text_area.configure(state='disabled')

        psf_plot_tab = ttk.Frame(output_notebook, padding="5")
        output_notebook.add(psf_plot_tab, text="PSF FWHM Plot")
        self.psf_plot_label = ttk.Label(psf_plot_tab, text="PSF FWHM vs. Position plot will appear here.\n(Scroll=Zoom, Drag=Pan)", relief="groove", anchor="center", justify="center")
        self.psf_plot_label.pack(fill=tk.BOTH, expand=True)
        self.psf_plot_label.bind("<MouseWheel>", self._on_mouse_wheel_psf_plot_gui)
        self.psf_plot_label.bind("<Button-4>", self._on_mouse_wheel_psf_plot_gui)
        self.psf_plot_label.bind("<Button-5>", self._on_mouse_wheel_psf_plot_gui)
        self.psf_plot_label.bind("<ButtonPress-1>", self._on_button_press_psf_plot_gui)
        self.psf_plot_label.bind("<B1-Motion>", self._on_mouse_drag_psf_plot_gui)
        self.psf_plot_label.bind("<ButtonRelease-1>", self._on_button_release_psf_plot_gui)

        self.status_var = tk.StringVar(); self.status_var.set("Ready. Select a file and configure parameters.")
        ttk.Label(master, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).pack(side=tk.BOTTOM, fill=tk.X, pady=(5,0))

        # --- Start Queue Pollers and Thread Checker ---
        self.master.after(100, self.process_log_queue_gui)
        self.master.after(100, self.process_image_display_queue_gui)
        self.master.after(100, self.process_results_queue_gui)
        self.master.after(100, self.process_psf_plot_queue_gui)
        self.master.after(250, self._check_analysis_thread_status) # Start the button re-enabler check

    # --- UI Creation Methods (_create_*_params_ui, _create_display_adjustments_ui) ---
    # --- [OMITTED FOR BREVITY - Same as previous response] ---
    def _create_deskew_params_ui(self, parent_frame):
        ttk.Label(parent_frame, text="XY Pixel Size (µm):").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(parent_frame, textvariable=self.param_deskew_dx_um, width=8).grid(row=0, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Label(parent_frame, text="Z Stage Step (µm):").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(parent_frame, textvariable=self.param_deskew_dz_um, width=8).grid(row=1, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Label(parent_frame, text="LS Angle (°):").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(parent_frame, textvariable=self.param_deskew_angle_deg, width=8).grid(row=2, column=1, sticky=tk.EW, padx=2, pady=2)
        flip_frame = ttk.Frame(parent_frame)
        ttk.Label(flip_frame, text="Flip Direction:").pack(side=tk.LEFT, padx=2)
        ttk.Radiobutton(flip_frame, text="+1", variable=self.param_deskew_flip_direction, value=1).pack(side=tk.LEFT)
        ttk.Radiobutton(flip_frame, text="-1", variable=self.param_deskew_flip_direction, value=-1).pack(side=tk.LEFT)
        flip_frame.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(parent_frame, text="Save Intermediate Sheared Img", variable=self.param_deskew_save_intermediate_shear).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(parent_frame, text="Save Final Deskew Img", variable=self.param_deskew_save).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(parent_frame, text="Show/Save Deskew Plots (MIP PNGs)", variable=self.param_deskew_show_plots).grid(row=6, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(parent_frame, text="Smooth Shear", variable=self.param_deskew_smooth_shear).grid(row=7, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Label(parent_frame, text="Smooth Shear Y Sigma:").grid(row=8, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(parent_frame, textvariable=self.param_deskew_smooth_sigma, width=8).grid(row=8, column=1, sticky=tk.EW, padx=2, pady=2)
        parent_frame.columnconfigure(1, weight=1)


    def _create_decorr_params_ui(self, parent_frame):
        ttk.Label(parent_frame, text="Units Label:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(parent_frame, textvariable=self.param_decorr_units_label, width=8).grid(row=0, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Checkbutton(parent_frame, text="Show/Save Decorrelation Plots", variable=self.param_decorr_show_plots).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=2)
        parent_frame.columnconfigure(1, weight=1)

    def _create_psf_params_ui(self, parent_frame):
        ttk.Label(parent_frame, text="Padding (px):").grid(row=0, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(parent_frame, textvariable=self.param_psf_padding_pixels, width=8).grid(row=0, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Label(parent_frame, text="ROI Radius (px):").grid(row=1, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(parent_frame, textvariable=self.param_psf_roi_radius_pixels, width=8).grid(row=1, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Label(parent_frame, text="Intensity Threshold:").grid(row=2, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(parent_frame, textvariable=self.param_psf_intensity_threshold, width=8).grid(row=2, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Label(parent_frame, text="Fit R² Threshold:").grid(row=3, column=0, sticky=tk.W, padx=2, pady=2)
        ttk.Entry(parent_frame, textvariable=self.param_psf_fit_quality_r2_threshold, width=8).grid(row=3, column=1, sticky=tk.EW, padx=2, pady=2)
        ttk.Checkbutton(parent_frame, text="Generate & Show PSF FWHM Plot", variable=self.param_psf_show_plots).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=2)
        ttk.Checkbutton(parent_frame, text="Save PSF Threshold Plot", variable=self.param_psf_show_threshold_plot).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=2)
        parent_frame.columnconfigure(1, weight=1)


    def _create_display_adjustments_ui(self, parent_frame):
        parent_frame.columnconfigure(1, weight=1)

        ttk.Label(parent_frame, text="Min Display:").grid(row=0, column=0, sticky=tk.W, padx=2, pady=1)
        self.min_display_slider = ttk.Scale(parent_frame, from_=0, to=65535, orient=tk.HORIZONTAL, variable=self.display_min_slider_var, length=150, command=self._on_display_sliders_change_gui)
        self.min_display_slider.grid(row=0, column=1, sticky=tk.EW, padx=2, pady=1)
        self.min_display_val_label = ttk.Label(parent_frame, text="0", width=7, anchor='w')
        self.min_display_val_label.grid(row=0, column=2, sticky=tk.W, padx=2)

        ttk.Label(parent_frame, text="Max Display:").grid(row=1, column=0, sticky=tk.W, padx=2, pady=1)
        self.max_display_slider = ttk.Scale(parent_frame, from_=0, to=65535, orient=tk.HORIZONTAL, variable=self.display_max_slider_var, length=150, command=self._on_display_sliders_change_gui)
        self.max_display_slider.grid(row=1, column=1, sticky=tk.EW, padx=2, pady=1)
        self.max_display_val_label = ttk.Label(parent_frame, text="255", width=7, anchor='w')
        self.max_display_val_label.grid(row=1, column=2, sticky=tk.W, padx=2)

        def update_min_label(*args): self.min_display_val_label.config(text=f"{self.display_min_slider_var.get():.0f}")
        self.display_min_slider_var.trace_add("write", update_min_label)
        def update_max_label(*args): self.max_display_val_label.config(text=f"{self.display_max_slider_var.get():.0f}")
        self.display_max_slider_var.trace_add("write", update_max_label)

        button_frame = ttk.Frame(parent_frame)
        button_frame.grid(row=2, column=0, columnspan=3, pady=(8,2), sticky="ew")
        button_frame.columnconfigure(0, weight=1)
        button_frame.columnconfigure(1, weight=1)

        self.auto_contrast_button = ttk.Button(button_frame, text="Auto", command=self._auto_adjust_contrast_gui, width=10)
        self.auto_contrast_button.grid(row=0, column=0, sticky='w', padx=2)
        self.reset_display_button = ttk.Button(button_frame, text="Reset", command=self._reset_display_params_gui, width=10)
        self.reset_display_button.grid(row=0, column=1, sticky='e', padx=2)

        ttk.Label(parent_frame, textvariable=self.display_actual_min_max_label_var).grid(row=3, column=0, columnspan=3, sticky=tk.W, pady=(8,2))


    # --- Action and Queue Processing Methods ---
    def select_file(self):
        path = filedialog.askopenfilename(title='Select 3D TIFF Z-stack file',
                                          filetypes=[('TIFF Files', '*.tif *.tiff'), ('All files', '*.*')])
        if path:
            self.filepath.set(path)
            self.status_var.set(f"Selected: {os.path.basename(path)}")
            self.log_text_area.configure(state='normal'); self.log_text_area.delete(1.0, tk.END); self.log_text_area.configure(state='disabled')
            self.results_text_area.configure(state='normal'); self.results_text_area.delete(1.0, tk.END); self.results_text_area.configure(state='disabled')
            self._clear_displayed_image_gui()
            self._clear_displayed_psf_plot_gui()

    def _clear_displayed_image_gui(self):
        self.image_label.configure(image='', text="Combined MIP from Deskewing will appear here")
        self.tk_image = None; self.original_pil_image = None; self.original_image_data_np = None
        self.current_zoom_level = 1.0; self.view_rect_on_original = None
        self.image_data_min_val = 0; self.image_data_max_val=255
        self.display_min_slider_var.set(0)
        self.display_max_slider_var.set(255)
        self.min_display_slider.config(from_=0, to=255)
        self.max_display_slider.config(from_=0, to=255)
        self.display_actual_min_max_label_var.set("Img Min/Max: N/A")

    def _clear_displayed_psf_plot_gui(self):
        self.psf_plot_label.configure(image='', text="PSF FWHM vs. Position plot will appear here.\n(Scroll=Zoom, Drag=Pan)")
        self.psf_tk_plot_image = None; self.psf_plot_original_pil_image = None
        self.psf_plot_current_zoom_level = 1.0; self.psf_plot_view_rect_on_original = None

    def run_analysis_thread_gui(self):
        initial_file_path = self.filepath.get()
        if not initial_file_path or not os.path.exists(initial_file_path):
            messagebox.showerror("Error", "Please select a valid TIFF file first.")
            return

        # !! Crucial Check: Only start if no thread is running !!
        if self.analysis_thread is not None and self.analysis_thread.is_alive():
            messagebox.showwarning("Busy", "An analysis is already in progress. Please wait.")
            return

        try:
            params = {
                "initial_file_path": initial_file_path,
                "DESKEW_DX_UM": self.param_deskew_dx_um.get(),
                "DESKEW_DZ_UM": self.param_deskew_dz_um.get(),
                "DESKEW_ANGLE_DEG": self.param_deskew_angle_deg.get(),
                "DESKEW_FLIP_DIRECTION": self.param_deskew_flip_direction.get(),
                "DESKEW_SAVE_INTERMEDIATE_SHEAR": self.param_deskew_save_intermediate_shear.get(),
                "SHOW_DESKEW_PLOTS": self.param_deskew_show_plots.get(),
                "DECORR_UNITS_LABEL": self.param_decorr_units_label.get(),
                "SHOW_DECORR_PLOTS": self.param_decorr_show_plots.get(),
                "PSF_PADDING_PIXELS": self.param_psf_padding_pixels.get(),
                "PSF_ROI_RADIUS_PIXELS": self.param_psf_roi_radius_pixels.get(),
                "PSF_INTENSITY_THRESHOLD": self.param_psf_intensity_threshold.get(),
                "PSF_FIT_QUALITY_R2_THRESHOLD": self.param_psf_fit_quality_r2_threshold.get(),
                "SHOW_PSF_PLOTS": self.param_psf_show_plots.get(),
                "SHOW_PSF_THRESHOLD_PLOT": self.param_psf_show_threshold_plot.get(),
                "DESKEW_SMOOTH_SHEAR":self.param_deskew_smooth_shear.get(),
                "DESKEW_SMOOTH_SIGMA": self.param_deskew_smooth_sigma.get(),
                "DESKEW_SAVE":self.param_deskew_save.get()
            }
            if not (0 < params["DESKEW_DX_UM"] < 100 and 0 < params["DESKEW_DZ_UM"] < 100 and \
                    0 <= params["DESKEW_ANGLE_DEG"] < 90):
                raise ValueError("Deskew dx, dz, or angle out of reasonable range.")
            if params["DESKEW_FLIP_DIRECTION"] not in [1, -1]:
                raise ValueError("Deskew Flip Direction must be 1 or -1.")

        except (ValueError, tk.TclError) as e:
            messagebox.showerror("Parameter Error", f"Invalid parameter value: {e}")
            return

        self.process_button.config(state=tk.DISABLED)
        self.status_var.set("Processing... Please wait.")

        self.log_text_area.configure(state='normal'); self.log_text_area.delete(1.0, tk.END); self.log_text_area.configure(state='disabled')
        self.results_text_area.configure(state='normal'); self.results_text_area.delete(1.0, tk.END); self.results_text_area.configure(state='disabled')
        self._clear_displayed_image_gui()
        self.image_label.configure(text="Processing...")
        self._clear_displayed_psf_plot_gui()

        # !! Assign the new thread object !!
        self.analysis_thread = threading.Thread(target=self._run_full_analysis_orchestrator,
                                                args=(params, self.log_queue, self.image_display_queue,
                                                      self.results_queue, self.psf_plot_path_queue))
        self.analysis_thread.daemon = True
        self.analysis_thread.start()
        # The periodic checker will handle re-enabling the button

    def _run_full_analysis_orchestrator(self, params, log_q, img_q, res_q, psf_plot_q):
        # ... (This function remains the same) ...
        try:
            log_message_gui("===== ANALYSIS PIPELINE STARTED =====", log_q, "IMPORTANT")

            deskewed_tiff_path = None
            deskew_output_folder = None
            original_file_name = os.path.basename(params["initial_file_path"])

            log_message_gui("\n\n===== STEP 1: DESKEWING =====\n", log_q)
            deskewed_tiff_path, final_dx_um, final_dz_um, deskew_output_folder, \
            file_name_from_deskew, combined_mip_display_path = \
                perform_deskewing(params["initial_file_path"], params["DESKEW_DX_UM"], params["DESKEW_DZ_UM"],
                                  params["DESKEW_ANGLE_DEG"], params["DESKEW_FLIP_DIRECTION"],
                                  params["DESKEW_SAVE_INTERMEDIATE_SHEAR"], params["SHOW_DESKEW_PLOTS"],
                                  log_q, apply_post_shear_smoothing=params["DESKEW_SMOOTH_SHEAR"], 
                                  smoothing_sigma_yc=params["DESKEW_SMOOTH_SIGMA"], save_final_deskew = params["DESKEW_SAVE"])

            original_file_name = file_name_from_deskew if file_name_from_deskew else original_file_name # Use returned name if available
            summary_results = ""

            if combined_mip_display_path:
                img_q.put(combined_mip_display_path)
            else:
                img_q.put(None) # Signal failure or skip

            if deskewed_tiff_path and deskew_output_folder:
                log_message_gui(f"\nDeskewing complete. Output: {deskewed_tiff_path}", log_q)
                log_message_gui(f"Effective pixel sizes: dx = {final_dx_um:.3f} um, dz' = {final_dz_um:.3f} um", log_q)
                summary_results += f"--- Deskewing Summary ---\n"
                summary_results += f"Output: {os.path.basename(deskewed_tiff_path)}\n"
                summary_results += f"Effective dx: {final_dx_um:.3f} um, dz': {final_dz_um:.3f} um\n"

                log_message_gui("\n\n===== STEP 2: DECORRELATION ANALYSIS =====\n", log_q)
                decorr_results = run_decorrelation_analysis(
                    deskewed_tiff_path,
                    stack_name_prefix=os.path.splitext(original_file_name)[0] + "_deskewed",
                    lateral_pixel_size_units=final_dx_um,
                    axial_pixel_size_units=final_dz_um,
                    units_label=params["DECORR_UNITS_LABEL"],
                    show_decorr_plots=params["SHOW_DECORR_PLOTS"],
                    log_queue=log_q,
                    main_output_folder=deskew_output_folder
                )

                summary_results += f"\n--- Decorrelation Analysis Summary ({params['DECORR_UNITS_LABEL']}) ---\n"
                if decorr_results:
                    for view, data in decorr_results.items():
                        res = data.get('resolution', 'N/A')
                        snr = data.get('SNR', 'N/A')
                        res_str = f"{res:.2f}" if isinstance(res, (int, float)) and np.isfinite(res) else str(res)
                        snr_str = f"{snr:.2f}" if isinstance(snr, (int, float)) and np.isfinite(snr) else str(snr)
                        log_message_gui(f"  {view}: Resolution = {res_str} {params['DECORR_UNITS_LABEL']}, SNR = {snr_str}", log_q)
                        summary_results += f"  {view}: Res = {res_str}, SNR = {snr_str}\n"
                else:
                    log_message_gui("Decorrelation analysis failed or produced no results.", log_q, "WARNING")
                    summary_results += "  Decorrelation analysis failed or produced no results.\n"


                log_message_gui("\n\n===== STEP 3: PSF FITTING ANALYSIS =====\n", log_q)
                psf_pixel_size_z_nm = final_dz_um * 1000.0
                psf_pixel_size_y_nm = final_dx_um * 1000.0
                psf_pixel_size_x_nm = final_dx_um * 1000.0

                psf_analysis_results = run_psf_fitting_analysis(
                    deskewed_tiff_path=deskewed_tiff_path,
                    base_file_name=os.path.splitext(original_file_name)[0] + "_deskewed",
                    pixel_size_z_nm=psf_pixel_size_z_nm,
                    pixel_size_y_nm=psf_pixel_size_y_nm,
                    pixel_size_x_nm=psf_pixel_size_x_nm,
                    padding_pixels=params["PSF_PADDING_PIXELS"],
                    roi_radius_pixels=params["PSF_ROI_RADIUS_PIXELS"],
                    intensity_threshold=params["PSF_INTENSITY_THRESHOLD"],
                    fit_quality_r2_threshold=params["PSF_FIT_QUALITY_R2_THRESHOLD"],
                    show_psf_plots=params["SHOW_PSF_PLOTS"],
                    show_psf_threshold_plot=params["SHOW_PSF_THRESHOLD_PLOT"],
                    log_queue=log_q,
                    main_output_folder=deskew_output_folder,
                    psf_plot_path_queue=psf_plot_q
                )

                summary_results += f"\n--- PSF Fitting Analysis Summary (nm) ---\n"
                if psf_analysis_results and "summary_stats" in psf_analysis_results:
                    summary = psf_analysis_results["summary_stats"]
                    log_message_gui(f"  Number of quality-filtered beads: {summary.get('num_good_beads', 'N/A')}", log_q)
                    log_message_gui(f"  Mean FWHM Z: {summary.get('mean_fwhm_z_nm', np.nan):.2f} ± {summary.get('std_fwhm_z_nm', np.nan):.2f} nm", log_q)
                    log_message_gui(f"  Mean FWHM Y: {summary.get('mean_fwhm_y_nm', np.nan):.2f} ± {summary.get('std_fwhm_y_nm', np.nan):.2f} nm", log_q)
                    log_message_gui(f"  Mean FWHM X: {summary.get('mean_fwhm_x_nm', np.nan):.2f} ± {summary.get('std_fwhm_x_nm', np.nan):.2f} nm", log_q)
                    summary_results += f"  Num good beads: {summary.get('num_good_beads', 'N/A')}\n"
                    summary_results += f"  Mean FWHM Z: {summary.get('mean_fwhm_z_nm', np.nan):.2f} ± {summary.get('std_fwhm_z_nm', np.nan):.2f}\n"
                    summary_results += f"  Mean FWHM Y: {summary.get('mean_fwhm_y_nm', np.nan):.2f} ± {summary.get('std_fwhm_y_nm', np.nan):.2f}\n"
                    summary_results += f"  Mean FWHM X: {summary.get('mean_fwhm_x_nm', np.nan):.2f} ± {summary.get('std_fwhm_x_nm', np.nan):.2f}\n"

                    if deskew_output_folder:
                        psf_results_savename = Path(deskew_output_folder) / f"{os.path.splitext(original_file_name)[0]}_psf_fitting_results.npz"
                        try:
                            save_dict = {}
                            for k, v in psf_analysis_results.items():
                                if isinstance(v, (np.ndarray, list, dict, str, int, float, bool, tuple)):
                                    if isinstance(v, dict):
                                        save_dict[k] = {sk: sv for sk, sv in v.items() if isinstance(sv, (np.ndarray, list, str, int, float, bool, tuple))}
                                    else:
                                         save_dict[k] = v
                                elif v is None:
                                    save_dict[k] = v
                                else:
                                    log_message_gui(f"Skipping non-serializable key '{k}' of type {type(v)} for NPZ save.", log_q, "WARNING")

                            np.savez(psf_results_savename, **save_dict)
                            log_message_gui(f"PSF fitting detailed results saved to: {psf_results_savename}", log_q)
                        except Exception as e_save:
                            log_message_gui(f"Could not save PSF fitting results: {e_save}", log_q, "ERROR")
                else:
                    log_message_gui("PSF fitting analysis failed or produced no results.", log_q, "WARNING")
                    summary_results += "  PSF fitting analysis failed or produced no results.\n"
                    if psf_plot_q: psf_plot_q.put(None) # Ensure PSF plot queue gets a signal
            else:
                log_message_gui("Deskewing failed. Subsequent analysis steps will be skipped.", log_q, "ERROR")
                summary_results += "Deskewing failed. Subsequent analysis skipped.\n"
                if psf_plot_q: psf_plot_q.put(None) # Ensure PSF plot queue gets a signal

            res_q.put(summary_results)
            log_message_gui("\n\n===== ANALYSIS PIPELINE FINISHED =====\n", log_q, "IMPORTANT")

        except Exception as e:
            error_full_traceback = traceback.format_exc()
            error_msg = f"Critical error in analysis pipeline: {e}\n{error_full_traceback}"
            log_message_gui(error_msg, log_q, "CRITICAL_ERROR")
            res_q.put(f"Pipeline CRASHED:\n{e}")
            img_q.put(None) # Signal no image
            if psf_plot_q: psf_plot_q.put(None) # Signal no plot
        finally:
            gc.collect()


    def process_log_queue_gui(self):
        try:
            while True: # Process all available messages
                msg = self.log_queue.get_nowait()
                self.log_text_area.configure(state='normal')
                self.log_text_area.insert(tk.END, msg)
                self.log_text_area.see(tk.END)
                self.log_text_area.configure(state='disabled')
        except queue.Empty:
            pass
        except Exception as e:
            print(f"CRITICAL ERROR in process_log_queue_gui: {e}\n{traceback.format_exc()}")
        finally:
            if hasattr(self, 'master') and self.master.winfo_exists():
                self.master.after(100, self.process_log_queue_gui)

    def process_image_display_queue_gui(self):
        try:
            image_path_to_display = self.image_display_queue.get_nowait()
            if image_path_to_display and os.path.exists(image_path_to_display):
                self._setup_initial_image_for_display_gui(image_path_to_display, autofit=True)
                # Set status only if it's still showing a processing message
                current_status = self.status_var.get()
                self.process_button.config(state=tk.NORMAL)
                if "Processing..." in current_status:
                     self.status_var.set("Processing complete. Combined MIP displayed.")
                     self.process_button.config(state=tk.NORMAL)
                     button_was_re_enabled = True

                     
                     
            elif image_path_to_display is None:
                 self._clear_displayed_image_gui()
                 self.image_label.configure(text="Deskew MIP generation failed or skipped.")
        except queue.Empty:
            pass
        except Exception as e:
            log_message_gui(f"GUI: Error in image display queue processing: {e}\n{traceback.format_exc()}", self.log_queue, "ERROR")
            self._clear_displayed_image_gui()
            self.image_label.configure(text="Error displaying image.")
            self.status_var.set("Error displaying image.")
        finally:
            # Schedule next check only if master window still exists
            if hasattr(self, 'master') and self.master.winfo_exists():
                self.master.after(200, self.process_image_display_queue_gui)


    def process_results_queue_gui(self):
        try:
            summary = self.results_queue.get_nowait()
            self.results_text_area.configure(state='normal')
            self.results_text_area.delete(1.0, tk.END)
            self.results_text_area.insert(tk.END, summary)
            self.results_text_area.configure(state='disabled')

            current_status = self.status_var.get()
            # Update status bar unless thread checker already set it to "Ready"
            if "Ready for next run" not in current_status:
                if "CRASHED" in summary or "failed" in summary.lower():
                    self.status_var.set("Analysis completed with errors or failures.")
                elif "MIP displayed" not in current_status: # Don't overwrite MIP success message
                    self.status_var.set("Analysis complete. Summary updated.")
                    self.process_button.config(state=tk.NORMAL)

        except queue.Empty:
            pass
        except Exception as e:
            log_message_gui(f"GUI: Error in results queue processing: {e}\n{traceback.format_exc()}", self.log_queue, "ERROR")
            self.results_text_area.configure(state='normal')
            self.results_text_area.delete(1.0, tk.END)
            self.results_text_area.insert(tk.END, f"Error displaying results: {e}")
            self.results_text_area.configure(state='disabled')
            self.status_var.set("Error displaying results.")
        finally:
            if hasattr(self, 'master') and self.master.winfo_exists():
                self.master.after(100, self.process_results_queue_gui)

    def process_psf_plot_queue_gui(self):
        try:
            plot_path = self.psf_plot_path_queue.get_nowait()
            if plot_path and os.path.exists(plot_path):
                self._setup_initial_psf_plot_for_display_gui(plot_path, autofit=True)
            elif plot_path is None:
                self._clear_displayed_psf_plot_gui()
                # Optionally set placeholder text:
                # self.psf_plot_label.configure(text="No PSF FWHM plot generated (e.g., no good beads).")
        except queue.Empty:
            pass
        except Exception as e:
            log_message_gui(f"GUI: Error in PSF plot queue processing: {e}\n{traceback.format_exc()}", self.log_queue, "ERROR")
            self._clear_displayed_psf_plot_gui()
            self.psf_plot_label.configure(text="Error displaying PSF plot.")
        finally:
            if hasattr(self, 'master') and self.master.winfo_exists():
                self.master.after(100, self.process_psf_plot_queue_gui)

    # --- Dedicated method to check thread status and re-enable button ---
    def _check_analysis_thread_status(self):
        try: # Add try-except around the whole check for robustness
            # Check if the attribute exists AND the thread object is not None AND it's no longer alive
            if hasattr(self, 'analysis_thread') and self.analysis_thread is not None and not self.analysis_thread.is_alive():
                button_was_re_enabled = False
                # Ensure button exists before configuring
                self.process_button.config(state=tk.NORMAL)
                if hasattr(self, 'process_button') and self.process_button.winfo_exists():
                     if self.process_button['state'] == tk.DISABLED:
                        self.process_button.config(state=tk.NORMAL)
                        button_was_re_enabled = True

                # Crucially, reset the thread attribute so a new run can start
                self.analysis_thread = None

                # Update status bar *only if* we just re-enabled the button
                # and it still shows processing (to avoid overwriting error messages)
                if button_was_re_enabled:
                    current_status = self.status_var.get()
                    if "Processing... Please wait." in current_status:
                         self.status_var.set("Analysis finished. Ready for next run.")
        except Exception as e:
            # Log error, but crucially reset state to allow next run
            print(f"Error in _check_analysis_thread_status: {e}\n{traceback.format_exc()}")
            if hasattr(self, 'process_button') and self.process_button['state'] == tk.DISABLED:
                try:
                    self.process_button.config(state=tk.NORMAL)
                except tk.TclError: pass # Ignore if widget destroyed
            self.analysis_thread = None
            if hasattr(self, 'status_var'):
                 try:
                     self.status_var.set("Error checking thread status. Ready for next run.")
                 except tk.TclError: pass # Ignore if widget destroyed
        finally:
            # Reschedule the check only if the master window still exists
            if hasattr(self, 'master') and self.master.winfo_exists():
                try:
                    self.master.after(250, self._check_analysis_thread_status)
                except tk.TclError:
                     # Window likely destroyed during the check logic
                     pass


    # --- Display Update and Interaction Methods ---
    # --- [OMITTED FOR BREVITY - Same as previous response] ---
    def _setup_initial_image_for_display_gui(self, image_path, autofit=False):
        try:
            log_message_gui(f"GUI: Loading combined MIP TIFF for display: {image_path}", self.log_queue)
            if not image_path.lower().endswith((".tif", ".tiff")):
                 self._handle_image_display_error_gui(f"Expected TIFF for display, got: {os.path.basename(image_path)}")
                 return

            self.original_pil_image = Image.open(image_path)
            self.original_pil_image.load() # Ensure data is loaded
            self.original_image_data_np = np.array(self.original_pil_image)


            if self.original_image_data_np.size > 0:
                 self.image_data_min_val = float(np.min(self.original_image_data_np))
                 self.image_data_max_val = float(np.max(self.original_image_data_np))
                 if self.image_data_max_val <= self.image_data_min_val:
                     self.image_data_max_val = self.image_data_min_val + 1

                 self.display_actual_min_max_label_var.set(f"Img Min/Max: {self.image_data_min_val:.0f}/{self.image_data_max_val:.0f}")

                 slider_min = self.image_data_min_val
                 slider_max = self.image_data_max_val
                 if slider_max <= slider_min: slider_max = slider_min + 1

                 self.min_display_slider.config(from_=slider_min, to=slider_max)
                 self.max_display_slider.config(from_=slider_min, to=slider_max)
                 # Set sliders AFTER configuring range
                 self.display_min_slider_var.set(slider_min)
                 self.display_max_slider_var.set(slider_max)
            else:
                raise ValueError("Loaded image array is empty for stats.")

            if autofit:
                self._autofit_image_to_label(self.original_pil_image, self.image_label,
                                             "current_zoom_level", "view_rect_on_original")
            else:
                self.current_zoom_level = 1.0
                self.view_rect_on_original = (0, 0, self.original_pil_image.width, self.original_pil_image.height)

            self._update_displayed_image_portion_gui()

        except Exception as e: self._handle_image_display_error_gui(f"Error loading initial image for GUI: {e}\n{traceback.format_exc()}")

    def _on_display_sliders_change_gui(self, *args):
        min_val = self.display_min_slider_var.get()
        max_val = self.display_max_slider_var.get()

        changed_var_name = args[0] if args else None

        if min_val > max_val:
            if changed_var_name == str(self.display_min_slider_var):
                self.display_max_slider_var.set(min_val)
            elif changed_var_name == str(self.display_max_slider_var):
                self.display_min_slider_var.set(max_val)
            else:
                 if self.display_min_slider_var.get() > self.display_max_slider_var.get() + 1e-6: # Check with tolerance
                    self.display_max_slider_var.set(self.display_min_slider_var.get())
                 else:
                    self.display_min_slider_var.set(self.display_max_slider_var.get())

        if self.original_pil_image: self._update_displayed_image_portion_gui()

    def _reset_display_params_gui(self):
        if self.original_image_data_np is not None and self.original_image_data_np.size > 0 :
            slider_min = self.image_data_min_val
            slider_max = self.image_data_max_val
            if slider_max <= slider_min: slider_max = slider_min + 1

            self.min_display_slider.config(from_=slider_min, to=slider_max)
            self.max_display_slider.config(from_=slider_min, to=slider_max)
            self.display_min_slider_var.set(slider_min)
            self.display_max_slider_var.set(slider_max)
        else:
            self.min_display_slider.config(from_=0, to=255)
            self.max_display_slider.config(from_=0, to=255)
            self.display_min_slider_var.set(0)
            self.display_max_slider_var.set(255)
        if self.original_pil_image: self._update_displayed_image_portion_gui()

    def _auto_adjust_contrast_gui(self):
        if self.original_image_data_np is None or self.view_rect_on_original is None:
            messagebox.showinfo("Auto Contrast", "No image loaded or view undefined.")
            return

        try:
            x0, y0, x1, y1 = map(int, self.view_rect_on_original)
            if x0 >= x1 or y0 >= y1 :
                log_message_gui("Auto Contrast: Invalid view rect for crop.", self.log_queue, "WARNING")
                if self.original_image_data_np.size > 0:
                    current_view_np = self.original_image_data_np
                else: return
            else:
                current_view_np = self.original_image_data_np[y0:y1, x0:x1]


            if current_view_np.size == 0:
                log_message_gui("Auto Contrast: Cropped view is empty.", self.log_queue, "WARNING")
                return

            finite_view_np = current_view_np[np.isfinite(current_view_np)]
            if finite_view_np.size == 0:
                log_message_gui("Auto Contrast: View contains no finite values.", self.log_queue, "WARNING")
                low_p = self.image_data_min_val
                high_p = self.image_data_max_val
            else:
                 try:
                     low_p, high_p = np.percentile(finite_view_np, (0.3, 99.7))
                 except IndexError:
                     low_p = np.min(finite_view_np)
                     high_p = np.max(finite_view_np)

            if high_p <= low_p:
                min_finite = np.min(finite_view_np) if finite_view_np.size > 0 else self.image_data_min_val
                max_finite = np.max(finite_view_np) if finite_view_np.size > 0 else self.image_data_max_val
                if max_finite > min_finite:
                    low_p = min_finite
                    high_p = max_finite
                else:
                    low_p = min_finite
                    high_p = min_finite + 1

            global_min_val_for_slider = self.min_display_slider.cget("from")
            global_max_val_for_slider = self.min_display_slider.cget("to")

            low_p = max(global_min_val_for_slider, float(low_p))
            high_p = min(global_max_val_for_slider, float(high_p))

            if high_p <= low_p:
                if high_p < global_max_val_for_slider: high_p = low_p + 1
                elif low_p > global_min_val_for_slider: low_p = high_p - 1

            self.display_min_slider_var.set(low_p)
            self.display_max_slider_var.set(high_p)

            log_message_gui(f"Auto Contrast: Set display range to [{low_p:.1f} - {high_p:.1f}]", self.log_queue)
            self._update_displayed_image_portion_gui()

        except Exception as e:
            log_message_gui(f"Error during auto contrast: {e}\n{traceback.format_exc()}", self.log_queue, "ERROR")
            messagebox.showerror("Auto Contrast Error", f"Could not auto-adjust: {e}")


    def _update_displayed_image_portion_gui(self):
        if self.original_image_data_np is None or self.view_rect_on_original is None:
            # Avoid clearing sliders if only data is missing temporarily
            # self._clear_displayed_image_gui();
            return
        try:
            self.image_label.update_idletasks()
            lbl_w, lbl_h = self.image_label.winfo_width(), self.image_label.winfo_height()
            if lbl_w <= 1 : lbl_w = max(200, int(self.master.winfo_width() * 0.4))
            if lbl_h <= 1 : lbl_h = max(200, int(self.master.winfo_height() * 0.4))

            crop_box_f = self.view_rect_on_original
            h_orig_np, w_orig_np = self.original_image_data_np.shape[:2]

            crop_x0 = max(0, int(crop_box_f[0]))
            crop_y0 = max(0, int(crop_box_f[1]))
            crop_x1 = min(w_orig_np, int(crop_box_f[2]))
            crop_y1 = min(h_orig_np, int(crop_box_f[3]))

            if crop_x0 >= crop_x1 or crop_y0 >= crop_y1:
                 self._handle_image_display_error_gui(f"Invalid crop region for NumPy: [{crop_y0}:{crop_y1}, {crop_x0}:{crop_x1}]"); return

            cropped_np_array = self.original_image_data_np[crop_y0:crop_y1, crop_x0:crop_x1]

            crop_h_np, crop_w_np = cropped_np_array.shape[:2]
            if crop_w_np == 0 or crop_h_np == 0: self._handle_image_display_error_gui("Cropped NumPy array has zero dimension."); return

            min_display = self.display_min_slider_var.get()
            max_display = self.display_max_slider_var.get()

            if min_display >= max_display:
                 # Ensure max_display is slightly larger if they are equal
                 max_display = min_display + 1e-9 # Add small epsilon

            # Use skimage.exposure.rescale_intensity - more robust than manual scaling
            # Need to handle potential non-finite values in the input array *before* rescaling if necessary
            # However, rescale_intensity should generally handle standard integer/float types
            input_array_for_rescale = cropped_np_array
            # Check if input is float and has NaNs/Infs
            if np.issubdtype(input_array_for_rescale.dtype, np.floating) and not np.all(np.isfinite(input_array_for_rescale)):
                log_message_gui("Warning: Input array for rescale contains non-finite values. Replacing with 0.", self.log_queue, "WARNING")
                input_array_for_rescale = np.nan_to_num(input_array_for_rescale, nan=0.0, posinf=max_display, neginf=min_display)

            adjusted_array_float = rescale_intensity(input_array_for_rescale,
                                                    in_range=(min_display, max_display),
                                                    out_range=(0.0, 255.0) # Output float range
                                                   )

            adjusted_array_8bit = np.clip(adjusted_array_float, 0, 255).astype(np.uint8)


            try: adjusted_img_pil_8bit = Image.fromarray(adjusted_array_8bit)
            except Exception as e: self._handle_image_display_error_gui(f"Error converting adjusted uint8->PIL: {e}"); return

            img_asp_adj = crop_w_np / crop_h_np if crop_h_np != 0 else float('inf')
            lbl_asp = lbl_w / lbl_h if lbl_h != 0 else float('inf')

            if img_asp_adj == float('inf') or lbl_asp == float('inf') or crop_w_np ==0 or crop_h_np == 0 or lbl_w == 0 or lbl_h == 0:
                new_w, new_h = lbl_w, lbl_h
            elif img_asp_adj > lbl_asp:
                new_w = lbl_w
                new_h = int(lbl_w / img_asp_adj) if img_asp_adj != 0 else 0
            else:
                new_h = lbl_h
                new_w = int(lbl_h * img_asp_adj)

            new_w,new_h = max(1,new_w),max(1,new_h)

            resized_img = adjusted_img_pil_8bit.resize((new_w, new_h), LANCZOS_RESAMPLE)
            new_photo = ImageTk.PhotoImage(resized_img)
            self.image_label.configure(image=new_photo, text="")
            self.tk_image = new_photo

        except Exception as e:
            self._handle_image_display_error_gui(f"Error updating display portion: {e}\n{traceback.format_exc()}")


    def _autofit_image_to_label(self, pil_image_obj, display_label_obj, zoom_attr_str, view_rect_attr_str):
        if pil_image_obj is None or display_label_obj is None: return

        display_label_obj.update_idletasks()
        lbl_w = display_label_obj.winfo_width()
        lbl_h = display_label_obj.winfo_height()

        # Use master geometry as fallback if label size is unreliable initially
        if lbl_w <= 1 : lbl_w = max(200, int(self.master.winfo_width() * 0.5))
        if lbl_h <= 1 : lbl_h = max(200, int(self.master.winfo_height() * 0.5))

        img_w, img_h = pil_image_obj.size
        if img_w == 0 or img_h == 0: return

        # Calculate the scale factor required to fit the image into the label
        scale_factor = 1.0
        if img_w > lbl_w or img_h > lbl_h: # Only scale down if image is larger than label
             scale_factor = min(lbl_w / img_w, lbl_h / img_h)

        # The conceptual zoom level (relative to 1:1 pixels) is this scale factor
        new_zoom = scale_factor

        # When autofitting, the view rectangle IS the whole image
        setattr(self, view_rect_attr_str, (0,0, img_w, img_h))
        # Store the calculated zoom level
        setattr(self, zoom_attr_str, new_zoom)

    def _on_mouse_wheel_gui(self, event):
        self._handle_mouse_wheel_for_image(event,
                                           self.original_pil_image,
                                           self.image_label,
                                           self.current_zoom_level,
                                           self.view_rect_on_original,
                                           self._update_displayed_image_portion_gui,
                                           "current_zoom_level",
                                           "view_rect_on_original")

    def _on_button_press_gui(self, event):
        self._handle_button_press_for_image(event,
                                            self.original_pil_image,
                                            self.image_label,
                                            self.view_rect_on_original,
                                            "pan_start_pos",
                                            "pan_start_view_offset")

    def _on_mouse_drag_gui(self, event):
        self._handle_mouse_drag_for_image(event,
                                           self.original_pil_image,
                                           self.image_label,
                                           self.pan_start_pos,
                                           self.view_rect_on_original,
                                           self.pan_start_view_offset,
                                           self._update_displayed_image_portion_gui,
                                           "view_rect_on_original")

    def _on_button_release_gui(self, event):
        self._handle_button_release_for_image(self.image_label,
                                              "pan_start_pos",
                                              "pan_start_view_offset")

    def _handle_image_display_error_gui(self, error_message):
        log_message_gui(f"GUI Image Display Error: {error_message}", self.log_queue, "ERROR")
        if hasattr(self, 'tk_image') and self.tk_image is not None: self.tk_image=None
        self.image_label.configure(text=f"Error displaying image.\nCheck logs.", image='');
        self.original_pil_image=None; self.original_image_data_np = None
        self.status_var.set("Error displaying image.")

    def _setup_initial_psf_plot_for_display_gui(self, image_path, autofit=False):
        try:
            log_message_gui(f"GUI: Loading PSF Plot PNG for display: {image_path}", self.log_queue)
            self.psf_plot_original_pil_image = Image.open(image_path)
            self.psf_plot_original_pil_image.load()

            if autofit:
                self._autofit_image_to_label(self.psf_plot_original_pil_image, self.psf_plot_label,
                                             "psf_plot_current_zoom_level", "psf_plot_view_rect_on_original")
            else:
                self.psf_plot_current_zoom_level = 1.0
                self.psf_plot_view_rect_on_original = (0, 0, self.psf_plot_original_pil_image.width, self.psf_plot_original_pil_image.height)

            self._update_displayed_psf_plot_portion_gui()
        except Exception as e:
            self._handle_psf_plot_display_error_gui(f"Error loading initial PSF plot: {e}\n{traceback.format_exc()}")

    def _update_displayed_psf_plot_portion_gui(self):
        if self.psf_plot_original_pil_image is None or self.psf_plot_view_rect_on_original is None:
            self._clear_displayed_psf_plot_gui(); return
        try:
            self.psf_plot_label.update_idletasks()
            lbl_w, lbl_h = self.psf_plot_label.winfo_width(), self.psf_plot_label.winfo_height()
            if lbl_w <= 1: lbl_w = max(200, int(self.master.winfo_width() * 0.4))
            if lbl_h <= 1: lbl_h = max(200, int(self.master.winfo_height() * 0.2))

            crop_box_f = self.psf_plot_view_rect_on_original
            crop_box_i = (max(0, int(crop_box_f[0])), max(0, int(crop_box_f[1])),
                          min(self.psf_plot_original_pil_image.width, int(crop_box_f[2])),
                          min(self.psf_plot_original_pil_image.height, int(crop_box_f[3])))

            if crop_box_i[0] >= crop_box_i[2] or crop_box_i[1] >= crop_box_i[3]:
                self._handle_psf_plot_display_error_gui(f"Invalid PSF plot crop region: {crop_box_i}"); return

            cropped_img_pil = self.psf_plot_original_pil_image.crop(crop_box_i)
            # Convert RGBA or RGB to RGB for PhotoImage compatibility if needed
            if cropped_img_pil.mode == 'RGBA':
                cropped_img_pil = cropped_img_pil.convert('RGB')

            crop_w, crop_h = cropped_img_pil.size
            if crop_w == 0 or crop_h == 0: self._handle_psf_plot_display_error_gui("Cropped PSF plot has zero dimension."); return

            img_asp_adj = crop_w / crop_h if crop_h != 0 else float('inf')
            lbl_asp = lbl_w / lbl_h if lbl_h != 0 else float('inf')

            if img_asp_adj == float('inf') or lbl_asp == float('inf') or crop_w ==0 or crop_h == 0 or lbl_w == 0 or lbl_h == 0:
                new_w, new_h = lbl_w, lbl_h
            elif img_asp_adj > lbl_asp:
                new_w = lbl_w
                new_h = int(lbl_w / img_asp_adj) if img_asp_adj != 0 else 0
            else:
                new_h = lbl_h
                new_w = int(lbl_h * img_asp_adj)

            new_w, new_h = max(1, new_w), max(1, new_h)

            resized_img = cropped_img_pil.resize((new_w, new_h), LANCZOS_RESAMPLE)
            new_photo = ImageTk.PhotoImage(resized_img)
            self.psf_plot_label.configure(image=new_photo, text="")
            self.psf_tk_plot_image = new_photo
        except Exception as e:
            self._handle_psf_plot_display_error_gui(f"Error updating PSF plot display: {e}\n{traceback.format_exc()}")

    def _on_mouse_wheel_psf_plot_gui(self, event):
        self._handle_mouse_wheel_for_image(event,
                                           self.psf_plot_original_pil_image,
                                           self.psf_plot_label,
                                           self.psf_plot_current_zoom_level,
                                           self.psf_plot_view_rect_on_original,
                                           self._update_displayed_psf_plot_portion_gui,
                                           "psf_plot_current_zoom_level",
                                           "psf_plot_view_rect_on_original")

    def _on_button_press_psf_plot_gui(self, event):
        self._handle_button_press_for_image(event,
                                            self.psf_plot_original_pil_image,
                                            self.psf_plot_label,
                                            self.psf_plot_view_rect_on_original,
                                            "psf_plot_pan_start_pos",
                                            "psf_plot_pan_start_view_offset")

    def _on_mouse_drag_psf_plot_gui(self, event):
        self._handle_mouse_drag_for_image(event,
                                           self.psf_plot_original_pil_image,
                                           self.psf_plot_label,
                                           self.psf_plot_pan_start_pos,
                                           self.psf_plot_view_rect_on_original,
                                           self.psf_plot_pan_start_view_offset,
                                           self._update_displayed_psf_plot_portion_gui,
                                           "psf_plot_view_rect_on_original")

    def _on_button_release_psf_plot_gui(self, event):
        self._handle_button_release_for_image(self.psf_plot_label,
                                              "psf_plot_pan_start_pos",
                                              "psf_plot_pan_start_view_offset")

    def _handle_psf_plot_display_error_gui(self, error_message):
        log_message_gui(f"GUI PSF Plot Display Error: {error_message}", self.log_queue, "ERROR")
        if hasattr(self, 'psf_tk_plot_image') and self.psf_tk_plot_image is not None: self.psf_tk_plot_image = None
        self.psf_plot_label.configure(text=f"Error displaying PSF plot.\nCheck logs.", image='');
        self.psf_plot_original_pil_image = None

    def _handle_mouse_wheel_for_image(self, event, original_image, display_label,
                                      current_zoom_level,
                                      view_rect,
                                      update_callback,
                                      zoom_attr_str, view_rect_attr_str):
        if original_image is None or view_rect is None: return

        if event.num == 5 or event.delta < 0: zoom_factor_actual = 1 / 1.2
        elif event.num == 4 or event.delta > 0: zoom_factor_actual = 1.2
        else: return

        new_conceptual_zoom = current_zoom_level * zoom_factor_actual
        new_conceptual_zoom = max(0.01, min(50.0, new_conceptual_zoom))
        if abs(new_conceptual_zoom - current_zoom_level) < 1e-6 : return

        x0_orig, y0_orig, x1_orig, y1_orig = view_rect
        display_label.update_idletasks()
        lbl_w, lbl_h = display_label.winfo_width(), display_label.winfo_height()
        if lbl_w <=1 or lbl_h <=1 : return

        mouse_x_lbl,mouse_y_lbl = event.x, event.y
        current_view_w_on_orig,current_view_h_on_orig = x1_orig-x0_orig, y1_orig-y0_orig
        if current_view_w_on_orig <= 0 or current_view_h_on_orig <= 0: return

        focus_x_on_orig = x0_orig + (mouse_x_lbl/lbl_w)*current_view_w_on_orig
        focus_y_on_orig = y0_orig + (mouse_y_lbl/lbl_h)*current_view_h_on_orig

        new_view_w_on_orig = original_image.width / new_conceptual_zoom
        new_view_h_on_orig = original_image.height / new_conceptual_zoom

        new_x0 = focus_x_on_orig - (mouse_x_lbl/lbl_w)*new_view_w_on_orig
        new_y0 = focus_y_on_orig - (mouse_y_lbl/lbl_h)*new_view_h_on_orig
        new_x1 = new_x0 + new_view_w_on_orig
        new_y1 = new_y0 + new_view_h_on_orig

        orig_img_w, orig_img_h = original_image.width, original_image.height

        # Clamp view to original image boundaries while trying to maintain center
        if new_x0 < 0:
            new_x1 -= new_x0
            new_x0 = 0
        if new_y0 < 0:
            new_y1 -= new_y0
            new_y0 = 0

        if new_x1 > orig_img_w:
            new_x0 -= (new_x1 - orig_img_w)
            new_x1 = orig_img_w
        if new_y1 > orig_img_h:
            new_y0 -= (new_y1 - orig_img_h)
            new_y1 = orig_img_h

        # Ensure final coords are within bounds after adjustment
        new_x0 = max(0, new_x0)
        new_y0 = max(0, new_y0)
        new_x1 = min(orig_img_w, new_x1)
        new_y1 = min(orig_img_h, new_y1)

        # Prevent zero or negative size
        if new_x1 <= new_x0: new_x1 = new_x0 + 1
        if new_y1 <= new_y0: new_y1 = new_y0 + 1
        # Re-clamp if adding 1 pushed it over
        new_x1 = min(orig_img_w, new_x1)
        new_y1 = min(orig_img_h, new_y1)

        setattr(self, zoom_attr_str, new_conceptual_zoom)
        setattr(self, view_rect_attr_str, (new_x0, new_y0, new_x1, new_y1))
        update_callback()


    def _handle_button_press_for_image(self, event, original_image, display_label,
                                       view_rect, pan_start_pos_attr_str, pan_start_offset_attr_str):
        if original_image is None or view_rect is None: return
        setattr(self, pan_start_pos_attr_str, (event.x,event.y))
        setattr(self, pan_start_offset_attr_str, (view_rect[0], view_rect[1]))
        display_label.config(cursor="fleur")

    def _handle_mouse_drag_for_image(self, event, original_image, display_label,
                                     pan_start_pos, view_rect, pan_start_view_offset,
                                     update_callback, view_rect_attr_str):
        if original_image is None or pan_start_pos is None or view_rect is None or pan_start_view_offset is None: return

        dx_scr,dy_scr = event.x-pan_start_pos[0], event.y-pan_start_pos[1]

        display_label.update_idletasks()
        lbl_w, lbl_h = display_label.winfo_width(), display_label.winfo_height()
        if lbl_w <=1 or lbl_h <=1 : return

        x0_curr_view,y0_curr_view,x1_curr_view,y1_curr_view = view_rect
        current_view_w_on_orig = max(1, x1_curr_view-x0_curr_view)
        current_view_h_on_orig = max(1, y1_curr_view-y0_curr_view)

        if lbl_w == 0 or lbl_h == 0: return

        px_orig_per_scr_x = current_view_w_on_orig / lbl_w
        px_orig_per_scr_y = current_view_h_on_orig / lbl_h

        dx_orig = dx_scr * px_orig_per_scr_x
        dy_orig = dy_scr * px_orig_per_scr_y

        new_x0_pan = pan_start_view_offset[0] - dx_orig
        new_y0_pan = pan_start_view_offset[1] - dy_orig

        orig_img_w, orig_img_h = original_image.width, original_image.height

        # Clamp the new top-left corner so the view stays within bounds
        new_x0_pan = max(0, min(new_x0_pan, orig_img_w - current_view_w_on_orig))
        new_y0_pan = max(0, min(new_y0_pan, orig_img_h - current_view_h_on_orig))

        new_x1_pan = new_x0_pan + current_view_w_on_orig
        new_y1_pan = new_y0_pan + current_view_h_on_orig

        setattr(self, view_rect_attr_str, (new_x0_pan, new_y0_pan, new_x1_pan, new_y1_pan))
        update_callback()

    def _handle_button_release_for_image(self, display_label, pan_start_pos_attr_str, pan_start_offset_attr_str):
        setattr(self, pan_start_pos_attr_str, None)
        setattr(self, pan_start_offset_attr_str, None)
        try: # Widget might be destroyed
            display_label.config(cursor="")
        except tk.TclError:
            pass
if __name__ == "__main__":
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if 'clam' in style.theme_names(): 
            style.theme_use('clam')
        elif 'aqua' in style.theme_names(): 
            style.theme_use('aqua')
        elif 'vista' in style.theme_names(): 
            style.theme_use('vista')
    except tk.TclError:
        print("TTK themes not available or error setting theme. Using default.")

    app = OPMAnalysisApp(root)
    root.mainloop()
   
